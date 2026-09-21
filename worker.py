import os
import cv2
import numpy as np
import hashlib
from celery import Celery
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from database_manager import atualizar_status_tarefa, registrar_face

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "processamento_facial",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

motor_arcface = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
motor_arcface.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.20)

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

        pasta_confianca = os.path.join(pasta_destino, "_descartados_confianca")
        pasta_tamanho = os.path.join(pasta_destino, "_descartados_tamanho")
        os.makedirs(pasta_confianca, exist_ok=True)
        os.makedirs(pasta_tamanho, exist_ok=True)

        nome_arquivo_original = os.path.basename(caminho_fisico)
        
        faces_arc = motor_arcface.get(img_bgr)
        combined = []
        
        for idx, face in enumerate(faces_arc):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            x1, y1 = max(0, x1), max(0, y1)
            
            if face.det_score < confianca_minima:
                crop_conf = img_bgr[y1:y2, x1:x2]
                if crop_conf.size > 0:
                    cv2.imwrite(os.path.join(pasta_confianca, f"{id_extracao}_arc_{idx}_{nome_arquivo_original}"), crop_conf)
                continue
                
            combined.append({"bbox": np.array([x1, y1, x2, y2]), "embedding": face.embedding.astype(np.float32), "source": "arcface"})

        yolo_boxes = _triagem_semantica(img_bgr)
        for idx, ybox in enumerate(yolo_boxes):
            if any(_iou(ybox, item["bbox"]) > 0.5 for item in combined):
                continue
                
            x1, y1, x2, y2 = ybox
            x1, y1 = max(0, x1), max(0, y1)
            crop = img_bgr[y1:y2, x1:x2]
            
            if crop.size > 0:
                extra = motor_arcface.get(crop)
                if extra:
                    face_extra = extra[0]
                    if face_extra.det_score >= confianca_minima:
                        combined.append({"bbox": np.array([x1, y1, x2, y2]), "embedding": face_extra.embedding.astype(np.float32), "source": "yolo"})
                    else:
                        cv2.imwrite(os.path.join(pasta_confianca, f"{id_extracao}_yolo_{idx}_{nome_arquivo_original}"), crop)

        combined = _dedup_detections(combined, iou_thr=0.5)
        combined = _dedup_por_embedding(combined, limite_distancia=0.15)

        if not combined:
            atualizar_status_tarefa(tarefa_id, "SEM_ROSTO")
            return "SEM_ROSTO_VALIDO"

        faces_validas = 0
        
        for idx, item in enumerate(combined, start=1):
            x1, y1, x2, y2 = item["bbox"]
            alt, larg = y2 - y1, x2 - x1
            
            if alt < tamanho_min_face or larg < tamanho_min_face:
                crop_peq = img_bgr[y1:y2, x1:x2]
                if crop_peq.size > 0:
                    caminho_peq = os.path.join(pasta_tamanho, f"{id_extracao}_face{idx}_{nome_arquivo_original}")
                    cv2.imwrite(caminho_peq, crop_peq)
                continue
            
            faces_validas += 1
            vetor_512d = item["embedding"]
            hash_face = hashlib.sha256(f"{hash_base}_face_{idx}".encode()).hexdigest()
            caminho_crop_rel = f"crop_virtual_{id_extracao}_{idx}.jpg"
            
            status, ind_id, _ = registrar_face(
                vetor_512d=vetor_512d,
                hash_foto=hash_face,
                id_extracao=id_extracao,
                caminho_original=caminho_fisico,
                caminho_crop=caminho_crop_rel,
                limite_distancia=tolerancia_match,
            )
            
            pasta_suspeito = os.path.join(pasta_destino, str(ind_id))
            os.makedirs(pasta_suspeito, exist_ok=True)
            cv2.imwrite(
                os.path.join(pasta_suspeito, f"celular_{id_extracao}_face_{idx}.jpg"),
                img_bgr[y1:y2, x1:x2],
            )

        if faces_validas == 0:
            atualizar_status_tarefa(tarefa_id, "SEM_ROSTO")
            return "SEM_ROSTO_VALIDO"

        atualizar_status_tarefa(tarefa_id, "CONCLUIDO")
        return f"CONCLUIDO: {faces_validas} faces"

    except Exception as e:
        atualizar_status_tarefa(tarefa_id, "FALHA")
        return f"ERRO: {str(e)}"