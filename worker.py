import cv2
import numpy as np
import hashlib
import os
from celery import Celery
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from database_manager import atualizar_status_tarefa, registrar_face, atualizar_caminho_crop_ocorrencia
from PIL import Image, ImageOps

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "processamento_facial",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

motor_arcface = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
motor_arcface.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.10)

yolo_model = YOLO("yolov8n-face.pt")

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

@app.task(bind=True)
def extrair_faces_tarefa(self, tarefa_id, id_extracao, caminho_fisico, pasta_destino, tolerancia_match, tamanho_min_face, confianca_minima):
    try:
        hasher = hashlib.sha256()
        with open(caminho_fisico, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        hash_base = hasher.hexdigest()

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
        limiar_corte = min(float(confianca_minima), 0.25)

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

        if not combined:
            results = yolo_model(img_bgr, verbose=False, conf=0.20)
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
                                fx1, fy1, fx2, fy2 = face_ex.bbox.astype(int)
                                abs_x1, abs_y1 = ax1 + max(0, fx1), ay1 + max(0, fy1)
                                abs_x2, abs_y2 = ax1 + max(0, fx2), ay1 + max(0, fy2)
                                
                                combined.append({
                                    "bbox": np.array([abs_x1, abs_y1, abs_x2, abs_y2]),
                                    "embedding": face_ex.embedding.astype(np.float32),
                                    "source": "yolo_arcface"
                                })

        if not combined:
            try:
                h_o, w_o = img_bgr.shape[:2]
                pad_h, pad_w = int(h_o * 0.35), int(w_o * 0.35)
                
                img_moldura = cv2.copyMakeBorder(
                    img_bgr, pad_h, pad_h, pad_w, pad_w, 
                    cv2.BORDER_CONSTANT, value=[128, 128, 128]
                )
                
                faces_moldura = motor_arcface.get(img_moldura)
                for face_m in faces_moldura:
                    if face_m.det_score >= 0.10:
                        bbox = face_m.bbox.astype(int)
                        rx1 = max(0, bbox[0] - pad_w)
                        ry1 = max(0, bbox[1] - pad_h)
                        rx2 = min(w_o, bbox[2] - pad_w)
                        ry2 = min(h_o, bbox[3] - pad_h)
                        
                        combined.append({
                            "bbox": np.array([rx1, ry1, rx2, ry2]),
                            "embedding": face_m.embedding.astype(np.float32),
                            "source": "resgate_moldura_artificial"
                        })
            except Exception:
                pass

        if not combined:
            try:
                img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img_clara = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                
                alt_orig, larg_orig = img_clara.shape[:2]
                img_zoom = cv2.resize(img_clara, (larg_orig * 2, alt_orig * 2), interpolation=cv2.INTER_CUBIC)
                
                rotacoes = [
                    (img_zoom, 0),
                    (cv2.rotate(img_zoom, cv2.ROTATE_90_CLOCKWISE), 90),
                    (cv2.rotate(img_zoom, cv2.ROTATE_90_COUNTERCLOCKWISE), 270)
                ]
                
                for img_rot, angulo in rotacoes:
                    faces_resgate = motor_arcface.get(img_rot)
                    if faces_resgate:
                        for face_res in faces_resgate:
                            if face_res.det_score >= 0.10:
                                bbox = face_res.bbox.astype(int)
                                bx1, by1, bx2, by2 = bbox[0] // 2, bbox[1] // 2, bbox[2] // 2, bbox[3] // 2
                                
                                if angulo == 90:
                                    x1, y1, x2, y2 = by1, larg_orig - bx2, by2, larg_orig - bx1
                                elif angulo == 270:
                                    x1, y1, x2, y2 = alt_orig - by2, bx1, alt_orig - by1, bx2
                                else:
                                    x1, y1, x2, y2 = bx1, by1, bx2, by2
                                    
                                combined.append({
                                    "bbox": np.array([max(0, x1), max(0, y1), max(0, x2), max(0, y2)]),
                                    "embedding": face_res.embedding.astype(np.float32),
                                    "source": f"resgate_optico_rot_{angulo}"
                                })
                        if combined:
                            break
            except Exception:
                pass

        if not combined:
            caminho_sr = os.path.join(pasta_sem_rosto, nome_arquivo_original)
            cv2.imwrite(caminho_sr, img_bgr)
            atualizar_status_tarefa(tarefa_id, "SEM_ROSTO")
            return "SEM_ROSTO_VALIDO"

        combined = _dedup_detections(combined, iou_thr=0.4)
        combined = _dedup_por_embedding(combined, limite_distancia=0.10)

        faces_validas = 0
        tamanho_corte_min = min(int(tamanho_min_face), 30)

        for idx, item in enumerate(combined, start=1):
            x1, y1, x2, y2 = item["bbox"]
            alt, larg = y2 - y1, x2 - x1

            if alt < tamanho_corte_min or larg < tamanho_corte_min:
                continue

            faces_validas += 1
            vetor_512d = item["embedding"]
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