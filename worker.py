import cv2
import numpy as np
import hashlib
import os
from celery import Celery
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from database_manager import atualizar_status_tarefa, registrar_face, atualizar_caminho_crop_ocorrencia

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "processamento_facial",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

motor_arcface = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
motor_arcface.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.15)

yolo_model = YOLO(r"C:\Users\saulo.borges\Downloads\facial-cientifica\yolov8n.pt")

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

def _triagem_semantica(img_bgr):
    results = yolo_model(img_bgr)
    boxes = []
    for r in results:
        for det in r.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
    return boxes

def _dedup_detections(detections, iou_thr=0.5):
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

        img_bgr = cv2.imread(caminho_fisico)
        if img_bgr is None:
            atualizar_status_tarefa(tarefa_id, "FALHA")
            return f"FALHA: {caminho_fisico}"

        pasta_sem_rosto = os.path.join(pasta_destino, "_descartados_sem_rosto")
        os.makedirs(pasta_sem_rosto, exist_ok=True)
        nome_arquivo_original = os.path.basename(caminho_fisico)

        combined = []

        faces_arc = motor_arcface.get(img_bgr)
        for idx, face in enumerate(faces_arc):
            if face.det_score >= confianca_minima:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
                combined.append({
                    "bbox": np.array([x1, y1, x2, y2]),
                    "embedding": face.embedding.astype(np.float32),
                    "source": "arcface_direto"
                })

        if not combined:
            results = yolo_model(img_bgr, verbose=False, classes=[0])
            for r in results:
                for det in r.boxes:
                    x1_y, y1_y, x2_y, y2_y = map(int, det.xyxy[0].tolist())
                    h_c, w_c = y2_y - y1_y, x2_y - x1_y
                    pad_x, pad_y = int(w_c * 0.15), int(h_c * 0.15)
                    
                    ay1 = max(0, y1_y - pad_y)
                    ay2 = min(img_bgr.shape[0], y2_y + pad_y)
                    ax1 = max(0, x1_y - pad_x)
                    ax2 = min(img_bgr.shape[1], x2_y + pad_x)
                    
                    crop_pessoa = img_bgr[ay1:ay2, ax1:ax2]
                    
                    if crop_pessoa.size > 0:
                        extra = motor_arcface.get(crop_pessoa)
                        if extra:
                            for face_ex in extra:
                                if face_ex.det_score >= (confianca_minima - 0.10):
                                    fx1, fy1, fx2, fy2 = face_ex.bbox.astype(int)
                                    abs_x1, abs_y1 = x1_y + max(0, fx1), y1_y + max(0, fy1)
                                    abs_x2, abs_y2 = x1_y + max(0, fx2), y1_y + max(0, fy2)
                                    
                                    combined.append({
                                        "bbox": np.array([abs_x1, abs_y1, abs_x2, abs_y2]),
                                        "embedding": face_ex.embedding.astype(np.float32),
                                        "source": "yolo_arcface"
                                    })
        if not combined:
            try:
                img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img_clara = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                
                alt_orig, larg_orig = img_clara.shape[:2]
                img_zoom = cv2.resize(img_clara, (larg_orig * 2, alt_orig * 2), interpolation=cv2.INTER_CUBIC)
                
                faces_resgate = motor_arcface.get(img_zoom)
                for face_res in faces_resgate:
                    if face_res.det_score >= 0.15:
                        bbox = face_res.bbox.astype(int)
                        hx = max(0, bbox[0] // 2)
                        hy = max(0, bbox[1] // 2)
                        hx2 = bbox[2] // 2
                        hy2 = bbox[3] // 2
                        
                        combined.append({
                            "bbox": np.array([hx, hy, hx2, hy2]),
                            "embedding": face_res.embedding.astype(np.float32),
                            "source": "resgate_optico"
                        })
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
        for idx, item in enumerate(combined, start=1):
            x1, y1, x2, y2 = item["bbox"]
            alt, larg = y2 - y1, x2 - x1

            if alt < tamanho_min_face or larg < tamanho_min_face:
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