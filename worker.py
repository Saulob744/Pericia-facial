import os
import cv2
import numpy as np
import hashlib
from celery import Celery
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from database_manager import atualizar_status_tarefa, registrar_face, atualizar_caminho_crop_ocorrencia
from PIL import Image, ImageOps

REDIS_URL = os.getenv("REDIS_URL", "redis://broker:6379/0")
app = Celery(
    "processamento_facial",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Inicialização dos motores
motor_arcface = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
motor_arcface.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)

yolo_model = YOLO("yolov8n-face.pt")

# Funções de Apoio e Deduplicação Biométrica

def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union != 0 else 0


def _dedup_por_embedding(detections, limite_distancia=0.15):
    filtrado = []
    for det in detections:
        emb1 = det["embedding"]
        duplicado = False
        for mantido in filtrado:
            emb2 = mantido["embedding"]
            n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
            if n1 > 0 and n2 > 0:
                dist = 1.0 - np.dot(emb1, emb2) / (n1 * n2)
                if dist < limite_distancia:
                    duplicado = True
                    break
        if not duplicado:
            filtrado.append(det)
    return filtrado


def _dedup_detections(detections, iou_thr=0.4):
    filtered = []
    for det in detections:
        if not any(_iou(det["bbox"], kept["bbox"]) > iou_thr for kept in filtered):
            filtered.append(det)
    return filtered


# Pipeline de Extração Facial Forense

def _processar_imagem_individual(item):
    """
    Executa a extração rigorosa de 1 imagem mantendo todas as etapas forenses.
    """
    tarefa_id = item.get("id") or item.get("tarefa_id")
    id_extracao = item.get("id_extracao")
    caminho_fisico = item.get("caminho_arquivo") or item.get("caminho_fisico")
    pasta_destino = item.get("pasta_destino", os.getenv("PASTA_DESTINO", "/app/data/armazenamento_crops"))
    tolerancia_match = item.get("tolerancia_match", 0.38)
    tamanho_min_face = item.get("tamanho_min_face", 30)
    confianca_minima = item.get("confianca_minima", 0.35)

    try:
        if not os.path.exists(caminho_fisico):
            atualizar_status_tarefa(tarefa_id, "ARQUIVO_NAO_ENCONTRADO")
            return "ARQUIVO_NAO_ENCONTRADO"

        # 1. Cálculo de HASH SHA256 do arquivo original
        hasher = hashlib.sha256()
        with open(caminho_fisico, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        hash_base = hasher.hexdigest()

        # 2. Leitura com correção EXIF
        try:
            pil_img = Image.open(caminho_fisico)
            pil_img = ImageOps.exif_transpose(pil_img)
            img_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        except Exception:
            img_bgr = cv2.imread(caminho_fisico)

        if img_bgr is None:
            atualizar_status_tarefa(tarefa_id, "FALHA")
            return f"FALHA: {caminho_fisico}"

        pasta_sem_rosto = os.path.join(pasta_destino, "_descartados_sem_rosto")
        os.makedirs(pasta_sem_rosto, exist_ok=True)
        nome_arquivo_original = os.path.basename(caminho_fisico)

        combined = []
        limiar_corte = min(float(confianca_minima), 0.35)

        # ETAPA A: ArcFace Direto
        faces_arc = motor_arcface.get(img_bgr)
        for idx, face in enumerate(faces_arc):
            if face.det_score >= limiar_corte:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
                combined.append({
                    "bbox": np.array([x1, y1, x2, y2]),
                    "embedding": face.embedding.astype(np.float32),
                    "source": "arcface_direto"
                })

        # ETAPA B: Resgate via YOLO + ArcFace (se ArcFace direto falhar)
        if not combined:
            results = yolo_model(img_bgr, verbose=False, conf=0.60)
            for r in results:
                for det in r.boxes:
                    x1_y, y1_y, x2_y, y2_y = map(int, det.xyxy[0].tolist())
                    h_c, w_c = y2_y - y1_y, x2_y - x1_y

                    pad_x, pad_y = int(w_c * 0.50), int(h_c * 0.50)

                    ay1 = max(0, y1_y - pad_y)
                    ay2 = min(img_bgr.shape[0], y2_y + pad_y)
                    ax1 = max(0, x1_y - pad_x)
                    ax2 = min(img_bgr.shape[1], x2_y + pad_x)

                    crop_pessoa = img_bgr[ay1:ay2, ax1:ax2]

                    if crop_pessoa.size > 0:
                        extra = motor_arcface.get(crop_pessoa)
                        if extra:
                            for face_ex in extra:
                                if face_ex.det_score >= 0.60:
                                    fx1, fy1, fx2, fy2 = face_ex.bbox.astype(int)
                                    abs_x1, abs_y1 = ax1 + max(0, fx1), ay1 + max(0, fy1)
                                    abs_x2, abs_y2 = ax1 + max(0, fx2), ay1 + max(0, fy2)

                                    combined.append({
                                        "bbox": np.array([abs_x1, abs_y1, abs_x2, abs_y2]),
                                        "embedding": face_ex.embedding.astype(np.float32),
                                        "source": "yolo_arcface"
                                    })

        # ETAPA C: Resgate com Moldura Artificial e Espelhamento (se ainda sem rostos)
        if not combined:
            try:
                h_o, w_o = img_bgr.shape[:2]
                pad_h, pad_w = int(h_o * 0.35), int(w_o * 0.35)

                img_moldura = cv2.copyMakeBorder(
                    img_bgr, pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT, value=[128, 128, 128]
                )

                variacoes_m = [(img_moldura, False), (cv2.flip(img_moldura, 1), True)]

                for img_m, foi_espelhada in variacoes_m:
                    faces_m = motor_arcface.get(img_m)
                    for face_m in faces_m:
                        if face_m.det_score >= 0.35:
                            bbox = face_m.bbox.astype(int)

                            if foi_espelhada:
                                w_m = img_m.shape[1]
                                bx1, bx2 = w_m - bbox[2], w_m - bbox[0]
                                by1, by2 = bbox[1], bbox[3]
                            else:
                                bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]

                            rx1 = max(0, bx1 - pad_w)
                            ry1 = max(0, by1 - pad_h)
                            rx2 = min(w_o, bx2 - pad_w)
                            ry2 = min(h_o, by2 - pad_h)

                            combined.append({
                                "bbox": np.array([rx1, ry1, rx2, ry2]),
                                "embedding": face_m.embedding.astype(np.float32),
                                "source": "resgate_moldura_artificial"
                            })
                    if combined:
                        break
            except Exception:
                pass

        # Descarte de Imagem Sem Rosto
        if not combined:
            caminho_sr = os.path.join(pasta_sem_rosto, nome_arquivo_original)
            cv2.imwrite(caminho_sr, img_bgr)
            atualizar_status_tarefa(tarefa_id, "SEM_ROSTO")
            return "SEM_ROSTO_VALIDO"

        # 3. Dupla Desduplicação (IoU + Embedding)
        combined = _dedup_detections(combined, iou_thr=0.4)
        combined = _dedup_por_embedding(combined, limite_distancia=0.20)

        faces_validas = 0
        tamanho_corte_min = min(int(tamanho_min_face), 30)

        # 4. Registro no Banco e Persistência dos Crops
        for idx, item_det in enumerate(combined, start=1):
            x1, y1, x2, y2 = item_det["bbox"]
            alt, larg = y2 - y1, x2 - x1

            if alt < tamanho_corte_min or larg < tamanho_corte_min:
                continue

            faces_validas += 1
            vetor_512d = item_det["embedding"]
            hash_face = hashlib.sha256(f"{hash_base}_{id_extracao}_face_{idx}".encode()).hexdigest()

            status, ind_id, _ = registrar_face(
                vetor_512d=vetor_512d,
                hash_foto=hash_face,
                id_extracao=id_extracao,
                caminho_original=caminho_fisico,
                caminho_crop="CAMINHO_PENDENTE",
                limite_distancia=tolerancia_match,
            )

            if status == "DUPLICADO" or ind_id is None:
                continue

            pasta_suspeito = os.path.join(pasta_destino, str(ind_id))
            os.makedirs(pasta_suspeito, exist_ok=True)

            nome_final_crop = f"celular_{id_extracao}_face_{idx}_{nome_arquivo_original}"
            caminho_final_absoluto = os.path.join(pasta_suspeito, nome_final_crop)

            cv2.imwrite(caminho_final_absoluto, img_bgr[y1:y2, x1:x2])
            atualizar_caminho_crop_ocorrencia(hash_face, caminho_final_absoluto)

        if faces_validas == 0:
            caminho_sr = os.path.join(pasta_sem_rosto, nome_arquivo_original)
            cv2.imwrite(caminho_sr, img_bgr)
            atualizar_status_tarefa(tarefa_id, "SEM_ROSTO")
            return "SEM_ROSTO_VALIDO"

        atualizar_status_tarefa(tarefa_id, "CONCLUIDO")
        return f"CONCLUIDO: {faces_validas} faces"

    except Exception as e:
        atualizar_status_tarefa(tarefa_id, "FALHA")
        return f"ERRO: {str(e)}"


# Tarefas Celery para Processamento Assíncrono

@app.task(bind=True, name="processar_lote_faces_tarefa")
def processar_lote_faces_tarefa(self, lote_tarefas):
    """
    Processa um lote de tarefas (vetorizado) de forma sequencial no worker.
    """
    if not lote_tarefas:
        return "LOTE_VAZIO"

    resultados = []
    for tarefa_item in lote_tarefas:
        res = _processar_imagem_individual(tarefa_item)
        resultados.append(res)

    return f"LOTE_CONCLUIDO: {len(resultados)} imagens"


@app.task(bind=True, name="extrair_faces_tarefa")
def extrair_faces_tarefa(self, tarefa_id, id_extracao, caminho_fisico, pasta_destino, tolerancia_match, tamanho_min_face, confianca_minima):
    """
    Mantido para compatibilidade legada (chamada individual 1 a 1).
    """
    item = {
        "tarefa_id": tarefa_id,
        "id_extracao": id_extracao,
        "caminho_fisico": caminho_fisico,
        "pasta_destino": pasta_destino,
        "tolerancia_match": tolerancia_match,
        "tamanho_min_face": tamanho_min_face,
        "confianca_minima": confianca_minima
    }
    return _processar_imagem_individual(item)