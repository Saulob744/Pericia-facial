import os
import psutil
import tracemalloc
import gc

# ==========================================
# A VACINA CONTRA O SEGFAULT (TF vs PYTORCH)
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import time
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURAÇÕES
# ==========================================
INPUT_DIR = "resultados_id_hunter/01_rostos_limpos"
OUTPUT_DIR = "resultados_id_hunter/bench_extracao"
METRICS_FILE = "resultados_id_hunter/metricas_extracao_hardware.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

# ==========================================
# INICIALIZAÇÃO DOS MODELOS
# ==========================================
print("="*60)
print("[INFO] CARREGANDO MODELOS NA MEMÓRIA...")

mtcnn_detector = None
yolo_detector = None
ssd_net = None

try:
    from ultralytics import YOLO
    yolo_detector = YOLO('yolov8n-face.pt')
    print("[OK] YOLOv8-Face Carregado.")
except Exception as e: print(f"[FALHA] YOLO: {e}")

try:
    from mtcnn import MTCNN
    mtcnn_detector = MTCNN()
    print("[OK] MTCNN Carregado.")
except Exception as e: print(f"[FALHA] MTCNN: {e}")

try:
    ssd_prototxt = "deploy.prototxt.txt"
    ssd_model = "res10_300x300_ssd_iter_140000.caffemodel"
    if os.path.exists(ssd_prototxt) and os.path.exists(ssd_model):
        ssd_net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_model)
        print("[OK] SSD Carregado.")
except Exception as e: print(f"[FALHA] SSD: {e}")

print("="*60)

# ==========================================
# FUNÇÕES BASE DE EXTRAÇÃO
# ==========================================
def run_mtcnn(img):
    return len(mtcnn_detector.detect_faces(img)) > 0

def run_yolo(img):
    return len(yolo_detector(img, verbose=False)[0].boxes) > 0

def run_ssd(img):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    ssd_net.setInput(blob)
    detections = ssd_net.forward()
    return (detections.shape[2] > 0 and detections[0, 0, 0, 2] > 0.5)

# ==========================================
# O "ESPIÃO" DE HARDWARE (PROFILER)
# ==========================================
def medir_consumo(funcao_alvo, img):
    # Força a limpeza do lixo da memória antes de medir para evitar ruídos
    gc.collect()
    
    processo = psutil.Process(os.getpid())
    
    # Captura o estado do PC antes do modelo agir
    memoria_antes_mb = processo.memory_info().rss / (1024 * 1024)
    cpu_tempo_antes = processo.cpu_times().user
    
    # Inicia o cronômetro e rastreador interno do Python
    tracemalloc.start()
    inicio = time.perf_counter()
    
    # EXECUTA O ALGORITMO
    sucesso = funcao_alvo(img)
    
    # Para tudo imediatamente
    tempo_decorrido_ms = (time.perf_counter() - inicio) * 1000
    _, pico_memoria_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Captura o estado do PC depois
    memoria_depois_mb = processo.memory_info().rss / (1024 * 1024)
    cpu_tempo_depois = processo.cpu_times().user
    
    # Cálculos Finais
    cpu_uso_ms = (cpu_tempo_depois - cpu_tempo_antes) * 1000
    
    # A RAM alocada em C++ (TensorFlow/PyTorch) reflete na diferença do Sistema Operacional
    ram_alocada_mb = max(0, memoria_depois_mb - memoria_antes_mb)
    if ram_alocada_mb == 0:
        ram_alocada_mb = pico_memoria_py / (1024 * 1024) # Fallback interno
        
    return sucesso, tempo_decorrido_ms, ram_alocada_mb, cpu_uso_ms

# ==========================================
# EXECUÇÃO DA ARENA V2
# ==========================================
arquivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:100]

print(f"\nIniciando Auditoria de Hardware com {len(arquivos)} imagens...")
metricas = []

for idx, arquivo in enumerate(arquivos):
    caminho = os.path.join(INPUT_DIR, arquivo)
    img_bgr = cv2.imread(caminho)
    if img_bgr is None: continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    m = {"Arquivo": arquivo}

    # Teste MTCNN
    if mtcnn_detector:
        ok, lat, ram, cpu = medir_consumo(run_mtcnn, img_rgb)
        m["MTCNN_Tempo_ms"], m["MTCNN_RAM_MB"], m["MTCNN_CPU_ms"] = lat, ram, cpu

    # Teste YOLO
    if yolo_detector:
        ok, lat, ram, cpu = medir_consumo(run_yolo, img_rgb)
        m["YOLO_Tempo_ms"], m["YOLO_RAM_MB"], m["YOLO_CPU_ms"] = lat, ram, cpu

    # Teste SSD
    if ssd_net:
        ok, lat, ram, cpu = medir_consumo(run_ssd, img_bgr)
        m["SSD_Tempo_ms"], m["SSD_RAM_MB"], m["SSD_CPU_ms"] = lat, ram, cpu

    metricas.append(m)
    if (idx + 1) % 10 == 0:
        print(f"-> Analisadas {idx + 1}/{len(arquivos)} imagens...")

# Salva os dados brutos
df = pd.DataFrame(metricas)
df.to_csv(METRICS_FILE, index=False)

# ==========================================
# RELATÓRIO FINAL
# ==========================================
print("\n" + "="*60)
print("AUDITORIA DE CONSUMO COMPUTACIONAL (MÉDIAS)")
print("="*60)

if mtcnn_detector:
    print(f"MTCNN -> Latência: {df['MTCNN_Tempo_ms'].mean():.2f} ms | RAM: {df['MTCNN_RAM_MB'].mean():.2f} MB | Uso CPU: {df['MTCNN_CPU_ms'].mean():.2f} ms")
if yolo_detector:
    print(f"YOLO  -> Latência: {df['YOLO_Tempo_ms'].mean():.2f} ms | RAM: {df['YOLO_RAM_MB'].mean():.2f} MB | Uso CPU: {df['YOLO_CPU_ms'].mean():.2f} ms")
if ssd_net:
    print(f"SSD   -> Latência: {df['SSD_Tempo_ms'].mean():.2f} ms | RAM: {df['SSD_RAM_MB'].mean():.2f} MB | Uso CPU: {df['SSD_CPU_ms'].mean():.2f} ms")
print("="*60)