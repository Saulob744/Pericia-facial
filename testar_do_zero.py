import os
import cv2
import hashlib
import warnings
import gc
import numpy as np
from insightface.app import FaceAnalysis
from database_manager import inserir_face, buscar_similar

warnings.filterwarnings('ignore', category=FutureWarning)

# ==========================================
# CONFIGURAÇÕES DE DIRETÓRIOS E REGRAS
# ==========================================
DIR_ENTRADA = "entradas_celular"
DIR_CROPS = "armazenamento_crops"
LIMIAR_SIMILARIDADE = 0.45 

os.makedirs(DIR_ENTRADA, exist_ok=True)
os.makedirs(DIR_CROPS, exist_ok=True)

def calcular_sha256(caminho_arquivo: str) -> str:
    hasher = hashlib.sha256()
    with open(caminho_arquivo, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def salvar_recorte_face(img, bbox, nome_arquivo, indice, nome_celular) -> str:
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    crop = img[y1:y2, x1:x2]
    caminho_crop = os.path.join(DIR_CROPS, f"crop_{nome_celular}_{indice}_{nome_arquivo}")
    cv2.imwrite(caminho_crop, crop)
    return caminho_crop

def executar_teste_real():
    print("=" * 65)
    print("[INICIALIZAÇÃO] Carregando modelo ArcFace (buffalo_l) na CPU...")
    print("=" * 65)

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
 
    app.prepare(ctx_id=0, det_size=(800, 800), det_thresh=0.25)

    pastas_celulares = [nome for nome in os.listdir(DIR_ENTRADA) if os.path.isdir(os.path.join(DIR_ENTRADA, nome))]
    
    if len(pastas_celulares) == 0:
        pastas_celulares = [""] 

    for celular in pastas_celulares:
        caminho_pasta_atual = os.path.join(DIR_ENTRADA, celular)
        nome_exibicao = celular if celular else "RAIZ"
        
        print("\n" + "=" * 65)
        print(f"📱 INICIANDO VARREDURA NO CELULAR / PASTA: {nome_exibicao}")
        print("=" * 65)

        arquivos = [f for f in os.listdir(caminho_pasta_atual) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(arquivos) == 0:
            print(f"  [AVISO] A pasta '{nome_exibicao}' está vazia ou sem imagens.")
            continue

        print(f"  [INFO] Encontradas {len(arquivos)} imagens em '{nome_exibicao}'.\n")

        for idx, arquivo in enumerate(arquivos, start=1):
            caminho_imagem = os.path.join(caminho_pasta_atual, arquivo)
            print(f"  [{idx}/{len(arquivos)}] Analisando: {arquivo}")

            hash_base = calcular_sha256(caminho_imagem)

            img = cv2.imread(caminho_imagem)
            if img is None:
                print(f"    └─ [ERRO] Imagem ilegível ou corrompida. Pulando...")
                continue

            faces = app.get(img)
            if len(faces) == 0:
                print(f"    └─ [AVISO] Nenhum rosto humano foi identificado nesta foto.")
                continue

            print(f"    └─ Detectada(s) {len(faces)} face(s) na imagem.")

            for f_idx, face in enumerate(faces, start=1):
                vetor_512d = face.embedding.astype(np.float32)

                hash_face_unica = hashlib.sha256(f"{hash_base}_face_{f_idx}".encode()).hexdigest()

                caminho_crop = salvar_recorte_face(img, face.bbox, arquivo, f"{idx}_{f_idx}", nome_exibicao)

                match = buscar_similar(vetor_512d, limite_distancia=LIMIAR_SIMILARIDADE)

                if match:
                    uuid_banco, origem_anterior, distancia = match
                    print(f"       🚨 [REINCIDÊNCIA DETECTADA]")
                    print(f"          ├─ Este rosto pertence ao ID: {uuid_banco}")
                    print(f"          ├─ Visto originalmente em: {origem_anterior}")
                    print(f"          └─ Distância matemática: {distancia:.4f} (<= {LIMIAR_SIMILARIDADE})")
                else:
                    print(f"       ✨ [NOVO INDIVÍDUO] Nenhuma correspondência anterior no banco.")

                novo_id = inserir_face(
                    hash_foto=hash_face_unica,
                    caminho_original=caminho_imagem,
                    caminho_crop=caminho_crop,
                    vetor_512d=vetor_512d
                )

                if novo_id:
                    print(f"          └─ Registrado no PostgreSQL com UUID: {novo_id}")
        
        gc.collect()
        print(f"\n✅ Varredura em '{nome_exibicao}' finalizada. Memória RAM liberada.")

if __name__ == "__main__":
    executar_teste_real()