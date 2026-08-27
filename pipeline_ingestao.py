import os
import gc
import cv2
import hashlib
import numpy as np
from insightface.app import FaceAnalysis
from database_manager import inserir_face, buscar_similar

# ==========================================
# CONFIGURAÇÕES DE DIRETÓRIO E PARÂMETROS
# ==========================================
INPUT_DIR = "resultados_id_hunter/01_rostos_limpos"
CROP_OUTPUT_DIR = "resultados_id_hunter/crops_recortados"

LIMIAR_SIMILARIDADE = 0.45

os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def calcular_sha256(caminho_arquivo: str) -> str:
    hasher = hashlib.sha256()
    with open(caminho_arquivo, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def salvar_recorte_face(img, bbox, nome_base, indice) -> str:
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    crop = img[y1:y2, x1:x2]
    caminho_crop = os.path.join(CROP_OUTPUT_DIR, f"crop_{indice}_{nome_base}")
    cv2.imwrite(caminho_crop, crop)
    return caminho_crop

# ==========================================
# MOTOR PRINCIPAL DE PROCESSAMENTO
# ==========================================
def executar_pipeline():
    print("=" * 60)
    print("[INICIALIZAÇÃO] Carregando Motor Biométrico ArcFace (Buffalo_L)...")
    print("=" * 60)

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    if not os.path.exists(INPUT_DIR):
        print(f"[ERRO] A pasta '{INPUT_DIR}' não existe. Crie a pasta e adicione as imagens.")
        return

    arquivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[INFO] {len(arquivos)} imagens encontradas para processamento.\n")

    contadores = {"processados": 0, "novos_inseridos": 0, "reincidentes_detectados": 0, "falhas": 0}

    for idx, arquivo in enumerate(arquivos, start=1):
        caminho_completo = os.path.join(INPUT_DIR, arquivo)
        print(f"[{idx}/{len(arquivos)}] Processando: {arquivo}")

        hash_arquivo = calcular_sha256(caminho_completo)

        img = cv2.imread(caminho_completo)
        if img is None:
            print(f"  └─ [AVISO] Imagem corrompida ou ilegível. Pulando...")
            contadores["falhas"] += 1
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"  └─ [AVISO] Nenhum rosto detectado pelo ArcFace. Pulando...")
            contadores["falhas"] += 1
            continue

        face_principal = faces[0]
        vetor_512d = face_principal.embedding.astype(np.float32)

        caminho_crop = salvar_recorte_face(img, face_principal.bbox, arquivo, idx)

        match = buscar_similar(vetor_512d, limite_distancia=LIMIAR_SIMILARIDADE)

        if match:
            uuid_existente, origem_existente, distancia = match
            print(f"  🚨 [REINCIDÊNCIA] Rosto idêntico/similar detectado no banco!")
            print(f"     ├─ ID no Banco: {uuid_existente}")
            print(f"     ├─ Origem Anterior: {origem_existente}")
            print(f"     └─ Distância Vetorial: {distancia:.4f}")
            contadores["reincidentes_detectados"] += 1
        else:
            print(f"  ✨ [NOVO INDIVÍDUO] Nenhuma correspondência encontrada. Cadastrando...")

        novo_id = inserir_face(
            hash_foto=hash_arquivo,
            caminho_original=caminho_completo,
            caminho_crop=caminho_crop,
            vetor_512d=vetor_512d
        )

        if novo_id:
            contadores["novos_inseridos"] += 1

        contadores["processados"] += 1

        if idx % 10 == 0:
            gc.collect()

    print("\n" + "=" * 60)
    print("RESUMO DA INGESTÃO BIOMÉTRICA:")
    print(f"- Fotos Processadas: {contadores['processados']}")
    print(f"- Novos Registros Gravados: {contadores['novos_inseridos']}")
    print(f"- Reincidências (Matches) Encontradas: {contadores['reincidentes_detectados']}")
    print(f"- Falhas / Sem Rosto: {contadores['falhas']}")
    print("=" * 60)

if __name__ == "__main__":
    executar_pipeline()