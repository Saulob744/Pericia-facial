import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

# ==========================================
# CONFIGURAÇÕES E DIRETÓRIOS
# ==========================================
INPUT_DIR = "resultados_id_hunter/01_rostos_limpos"
OUTPUT_DIR = "resultados_id_hunter/identificacao"
VETORES_FILE = os.path.join(OUTPUT_DIR, "vetores_512d.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# INICIALIZAÇÃO DO MOTOR BIOMÉTRICO (ARCFACE)
# ==========================================
print("="*50)
print("[INFO] A inicializar o motor ArcFace (InsightFace)...")
print("[AVISO] Na primeira execução, ele irá descarregar o modelo (~330MB).")
print("="*50)

# Inicializamos o modelo "buffalo_l" forçando o uso da CPU (sem GPU)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ==========================================
# FASE 3A: VETORIZAÇÃO (EXTRAÇÃO DE EMBEDDINGS)
# ==========================================
arquivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"\n[INFO] A vetorizar {len(arquivos)} rostos...")

lista_vetores = []
nomes_arquivos = []

for arquivo in arquivos:
    caminho = os.path.join(INPUT_DIR, arquivo)
    img = cv2.imread(caminho)
    
    if img is None: continue
    
    # O InsightFace extrai a face, alinha os olhos e gera o vetor matemático
    faces = app.get(img)
    
    if len(faces) > 0:
        # Pega na face principal e extrai as 512 dimensões
        vetor_512d = faces[0].embedding
        lista_vetores.append(vetor_512d)
        nomes_arquivos.append(arquivo)

# Transforma a lista de vetores numa matriz matemática (Numpy)
matriz_embeddings = np.array(lista_vetores)

# Guarda o CSV como prova técnica para a dissertação
df_vetores = pd.DataFrame(matriz_embeddings)
df_vetores.insert(0, "Arquivo", nomes_arquivos)
df_vetores.to_csv(VETORES_FILE, index=False)
print(f"[OK] Vetores matemáticos guardados em: {VETORES_FILE}")

# ==========================================
# FASE 3B: CLUSTERIZAÇÃO COM DBSCAN
# ==========================================
print("\n" + "="*50)
print("A INICIAR MOTOR DE RECONHECIMENTO (DBSCAN)")
print("="*50)

# 1. O "Truque" da Normalização L2 (Para a Distância Euclidiana simular o Cosseno)
embeddings_normalizados = normalize(matriz_embeddings, norm='l2')

# 2. Configuração do DBSCAN
# eps = 0.4 a 0.6 é o padrão ideal para embeddings normalizados do ArcFace
EPSILON = 0.5
MIN_SAMPLES = 1 # Definimos como 1 para que rostos únicos fiquem no seu próprio "cluster" (não sejam lixo)

clusterizador = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric='euclidean')
rotulos = clusterizador.fit_predict(embeddings_normalizados)

# 3. Análise dos Resultados
df_resultados = pd.DataFrame({
    "Arquivo": nomes_arquivos,
    "ID_Pessoa": rotulos
})

total_pessoas_unicas = len(set(rotulos)) - (1 if -1 in rotulos else 0)

print(f"[RESULTADO] O ID-Hunter analisou {len(nomes_arquivos)} fotos válidas.")
print(f"[RESULTADO] Foram identificadas {total_pessoas_unicas} PESSOAS ÚNICAS na base de dados.")
print("\n[Amostra de Identificação]:")
print(df_resultados.sort_values(by="ID_Pessoa").head(15).to_string(index=False))