import os
import time
import cv2
import hashlib
import gc
import numpy as np
import warnings
from insightface.app import FaceAnalysis
import argparse

from database_manager import pegar_tarefa_pendente, atualizar_status_tarefa, registrar_face

warnings.filterwarnings('ignore', category=FutureWarning)

def calcular_sha256(caminho_arquivo: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(caminho_arquivo, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def iniciar_worker(diretorio_raiz: str, diretorio_destino: str):
    print("=" * 70)
    print(f"👷 WORKER INICIADO: Lendo base física em '{diretorio_raiz}'")
    print(f"💾 DESTINO DOS CROPS: '{diretorio_destino}'")
    print("=" * 70)
    
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])

    app.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.25)
    
    cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    print("[INFO] Motor Biométrico Híbrido HD (1024px + Fallback OpenCV) inicializado com SUCESSO.\n")

    while True:
        tarefa = pegar_tarefa_pendente()

        if not tarefa:
            print("💤 Fila vazia. O Worker está descansando...", end="\r")
            time.sleep(5)
            continue

        tarefa_id, id_extracao, caminho_relativo_do_banco = tarefa
        caminho_fisico_real = os.path.join(diretorio_raiz, caminho_relativo_do_banco)

        print("\n" + "-" * 70)
        print(f"⏳ [PROCESSANDO TAREFA {tarefa_id}] | Celular: {id_extracao}")
        print(f"📂 Arquivo: {caminho_relativo_do_banco}")

        hash_base = calcular_sha256(caminho_fisico_real)
        if not hash_base:
            print("   └─ ❌ [ERRO] Arquivo físico não encontrado ou corrompido.")
            atualizar_status_tarefa(tarefa_id, 'FALHA')
            continue

        img = cv2.imread(caminho_fisico_real)
        if img is None:
            print("   └─ ❌ [ERRO] O OpenCV não conseguiu ler a imagem (formato inválido).")
            atualizar_status_tarefa(tarefa_id, 'FALHA')
            continue

        faces = app.get(img)
        
        usou_fallback = False
        if len(faces) == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(rects) == 0:
                rects = cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            if len(rects) > 0:
                usou_fallback = True
                print(f"   ├─ 🛡️ [REDE DE SEGURANÇA OPENCV] Recuperou {len(rects)} rosto(s) que seriam ignorados!")
                for (x, y, w, h) in rects:
                    pad = int(max(w, h) * 0.25)
                    x1_pad, y1_pad = max(0, x - pad), max(0, y - pad)
                    x2_pad, y2_pad = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
                    sub_crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                    if sub_crop.size > 0:
                        sub_faces = app.get(sub_crop)
                        if len(sub_faces) > 0:
                            faces.extend(sub_faces)

        if len(faces) == 0:
            print("   └─ 👻 [IGNORADO] Nenhum rosto encontrado mesmo com motor híbrido.")
            atualizar_status_tarefa(tarefa_id, 'SEM_ROSTO')
            continue

        print(f"   ├─ 🧠 [IA] Encontrou {len(faces)} face(s) válida(s). Filtrando biometria...")

        for f_idx, face in enumerate(faces, start=1):
            bbox = face.bbox.astype(int)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
            
            altura_rosto = y2 - y1
            largura_rosto = x2 - x1
            proporcao = largura_rosto / float(altura_rosto) if altura_rosto > 0 else 0
            
            if altura_rosto < 30 or largura_rosto < 30 or proporcao < 0.35 or proporcao > 2.5:
                print(f"   │")
                print(f"   ├─ [Rosto {f_idx}/{len(faces)}]")
                print(f"   │  🗑️ DESCARTADO: Dimensão ({largura_rosto}x{altura_rosto}px) insuficiente.")
                continue


                
            rosto_recortado = img[y1:y2, x1:x2]
            
            vetor_512d = face.embedding.astype(np.float32)
            hash_face_unica = hashlib.sha256(f"{hash_base}_face_{f_idx}".encode()).hexdigest()
            caminho_crop = f"crop_virtual_{id_extracao}_{f_idx}.jpg"

            status, individuo_id, historico_lotes = registrar_face(
                vetor_512d=vetor_512d, 
                hash_foto=hash_face_unica, 
                id_extracao=id_extracao, 
                caminho_original=caminho_relativo_do_banco,
                caminho_crop=caminho_crop
            )
            
            pasta_do_suspeito = os.path.join(diretorio_destino, str(individuo_id))
            os.makedirs(pasta_do_suspeito, exist_ok=True)
            
            nome_arquivo_crop = f"celular_{id_extracao}_face_{f_idx}.jpg"
            caminho_final_crop = os.path.join(pasta_do_suspeito, nome_arquivo_crop)
            
            if rosto_recortado.size > 0:
                cv2.imwrite(caminho_final_crop, rosto_recortado)

            print(f"   │")
            print(f"   ├─ [Rosto {f_idx}/{len(faces)}]")
            
            if status == "NOVO":
                print(f"   │  ✨ NOVO: Indivíduo inédito cadastrado! (ID: {individuo_id})")
                print(f"   │     └─ 📸 Rosto salvo em: {pasta_do_suspeito}")
                
            elif status == "REINCIDENTE":
                print(f"   │  🚨 MATCH: Sujeito já conhecido na base! (ID: {individuo_id})")
                if historico_lotes:
                    lotes_formatados = ", ".join(historico_lotes)
                    print(f"   │     └─ 🔗 Já visto nos lotes: [ {lotes_formatados} ]")
                    print(f"   │     └─ 📍 Agora também no lote: {id_extracao}")
                    print(f"   │     └─ 📸 Novo rosto salvo junto aos outros na pasta!")
                
            else:
                print(f"   │  ⚠️ IGNORADO: Rosto exato duplicado (já lido anteriormente).")

        print("   │")
        print("   └─ ✅ [CONCLUÍDO] Tarefa finalizada com sucesso no banco de dados e HD.")
        
        atualizar_status_tarefa(tarefa_id, 'CONCLUIDO')
        
        del img
        del faces
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operário de Biometria")
    parser.add_argument("--origem", type=str, required=True, help="Caminho raiz das fotos")
    parser.add_argument("--destino", type=str, required=True, help="Caminho onde salvar os crops")
    args = parser.parse_args()
    
    iniciar_worker(args.origem, args.destino)