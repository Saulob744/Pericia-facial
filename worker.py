import os
import cv2
import numpy as np
import hashlib
from celery import Celery
from insightface.app import FaceAnalysis
from database_manager import atualizar_status_tarefa, registrar_face

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    'processamento_facial',
    broker=REDIS_URL,
    backend=REDIS_URL
)

motor_arcface = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
motor_arcface.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.20)

@app.task(bind=True)
def extrair_faces_tarefa(self, tarefa_id, id_extracao, caminho_fisico, pasta_destino, tolerancia_match, tamanho_min_face):
    try:
        hasher = hashlib.sha256()
        with open(caminho_fisico, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        hash_base = hasher.hexdigest()

        img_bgr = cv2.imread(caminho_fisico)
        if img_bgr is None:
            atualizar_status_tarefa(tarefa_id, 'FALHA')
            return f"FALHA: {caminho_fisico}"

        pasta_sem_rosto = os.path.join(pasta_destino, "_descartados_sem_rosto")
        pasta_pequenos = os.path.join(pasta_destino, "_descartados_muito_pequenos")
        os.makedirs(pasta_sem_rosto, exist_ok=True)
        os.makedirs(pasta_pequenos, exist_ok=True)

        faces = motor_arcface.get(img_bgr)
        nome_arquivo_original = os.path.basename(caminho_fisico)

        if len(faces) == 0:
            caminho_lixo = os.path.join(pasta_sem_rosto, f"{id_extracao}_{nome_arquivo_original}")
            cv2.imwrite(caminho_lixo, img_bgr)
            
            atualizar_status_tarefa(tarefa_id, 'SEM_ROSTO')
            return "SEM_ROSTO"

        faces_validas = 0

        for f_idx, face in enumerate(faces, start=1):
            bbox = face.bbox.astype(int)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(img_bgr.shape[1], bbox[2]), min(img_bgr.shape[0], bbox[3])
            
            alt, larg = y2 - y1, x2 - x1
            
            if alt < tamanho_min_face or larg < tamanho_min_face:
                crop_pequeno = img_bgr[y1:y2, x1:x2]
                if crop_pequeno.size > 0:
                    caminho_lixo_pequeno = os.path.join(pasta_pequenos, f"{id_extracao}_face{f_idx}_{nome_arquivo_original}")
                    cv2.imwrite(caminho_lixo_pequeno, crop_pequeno)
                continue 
            
            faces_validas += 1
            vetor_512d = face.embedding.astype(np.float32)
            hash_face = hashlib.sha256(f"{hash_base}_face_{f_idx}".encode()).hexdigest()
            caminho_crop_rel = f"crop_virtual_{id_extracao}_{f_idx}.jpg"
            
            status, ind_id, _ = registrar_face(
                vetor_512d=vetor_512d, hash_foto=hash_face,
                id_extracao=id_extracao, caminho_original=caminho_fisico,
                caminho_crop=caminho_crop_rel, limite_distancia=tolerancia_match
            )

            pasta_suspeito = os.path.join(pasta_destino, str(ind_id))
            os.makedirs(pasta_suspeito, exist_ok=True)
            cv2.imwrite(os.path.join(pasta_suspeito, f"celular_{id_extracao}_face_{f_idx}.jpg"), img_bgr[y1:y2, x1:x2])

        if faces_validas == 0:
            atualizar_status_tarefa(tarefa_id, 'SEM_ROSTO')
            return "SEM_ROSTO_VALIDO"

        atualizar_status_tarefa(tarefa_id, 'CONCLUIDO')
        return f"CONCLUIDO: {faces_validas} faces"

    except Exception as e:
        atualizar_status_tarefa(tarefa_id, 'FALHA')
        return f"ERRO: {str(e)}"