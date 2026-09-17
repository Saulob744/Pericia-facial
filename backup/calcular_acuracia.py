import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS
# ==========================================
CAMINHO_MODELO = "best.pt" 
PASTA_BASE_TESTE = "dataset_teste"

print("[INFO] Carregando o modelo YOLOv8...")
model = YOLO(CAMINHO_MODELO)
nomes_classes_yolo = model.names
print(f"[INFO] Classes aprendidas pelo modelo: {nomes_classes_yolo}")

y_true = []
y_pred = []

extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ==========================================
# 2. INFERÊNCIA E COMPARAÇÃO
# ==========================================
print("\n[INFO] Iniciando validação nas imagens...")

for classe_real in os.listdir(PASTA_BASE_TESTE):
    pasta_classe = os.path.join(PASTA_BASE_TESTE, classe_real)
    
    if not os.path.isdir(pasta_classe):
        continue
        
    classe_real_normalizada = classe_real.lower()
    if classe_real_normalizada == "limpo":
        classe_real_normalizada = "face_limpa"
        
    for imagem in os.listdir(pasta_classe):
        if not imagem.lower().endswith(extensoes_validas):
            print(f"[AVISO] Ignorando arquivo não reconhecido: {imagem}")
            continue
            
        caminho_imagem = os.path.join(pasta_classe, imagem)
        
        resultados = model(caminho_imagem, verbose=False)
        classe_predita = "indefinido"
        
        if hasattr(resultados[0], 'probs') and resultados[0].probs is not None:
            id_predito = resultados[0].probs.top1
            classe_predita = model.names[id_predito]
            
        elif hasattr(resultados[0], 'boxes') and len(resultados[0].boxes) > 0:
            id_predito = int(resultados[0].boxes.cls[0])
            classe_predita = model.names[id_predito]
            
        else:
            classe_predita = "nao_detectado"
            
        y_true.append(classe_real_normalizada)
        y_pred.append(classe_predita.lower())

# ==========================================
# 3. CÁLCULOS METODOLÓGICOS E GRÁFICOS
# ==========================================
if len(y_true) == 0:
    print("\n[ERRO] Nenhuma imagem válida foi encontrada nas pastas para calcular a acurácia.")
else:
    acuracia = accuracy_score(y_true, y_pred)
    print("="*50)
    print(f"ACURÁCIA REAL ALCANÇADA: {acuracia * 100:.2f}%")
    print("="*50)

    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.title(f'Matriz de Confusão (YOLOv8-n)\nAcurácia Global: {acuracia*100:.2f}%', pad=20, fontsize=14)
    plt.ylabel('Classe Real (Ground Truth)', fontsize=12)
    plt.xlabel('Previsão do Modelo', fontsize=12)

    plt.savefig('matriz_confusao_academica.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráfico 'matriz_confusao_academica.png' salvo com sucesso na pasta do projeto!")