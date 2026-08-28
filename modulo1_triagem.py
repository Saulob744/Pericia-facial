import os
import cv2
import shutil
import hashlib
from PIL import Image
import imagehash
from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES E DIRETÓRIOS
# ==========================================
INPUT_DIR = "dados_brutos" # Pasta onde você vai jogar todas as fotos misturadas
BASE_OUTPUT_DIR = "resultados_id_hunter"

DIRS = {
    "limpos": os.path.join(BASE_OUTPUT_DIR, "01_rostos_limpos"),
    "oclusao": os.path.join(BASE_OUTPUT_DIR, "02_rostos_oclusao"),
    "lixo": os.path.join(BASE_OUTPUT_DIR, "03_descartes_lixo"),
    "duplicados": os.path.join(BASE_OUTPUT_DIR, "04_duplicados")
}

# Distância de Hamming aceitável (0 = idêntico, <= 3 = muito similar/redimensionado)
THRESHOLD_HAMMING = 3

# ==========================================
# CLASSES DE SERVIÇO (PIPELINE)
# ==========================================

class Deduplicador:
    """Responsável por garantir que a SESP não processe a mesma foto duas vezes."""
    def __init__(self):
        self.hashes_sha256 = set()
        self.hashes_perceptuais = []

    def calcular_sha256(self, filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def calcular_phash(self, filepath):
        try:
            img = Image.open(filepath)
            return imagehash.phash(img)
        except Exception:
            return None

    def eh_duplicado(self, filepath):
        # 1. Checagem Criptográfica (Exata)
        sha_hash = self.calcular_sha256(filepath)
        if sha_hash in self.hashes_sha256:
            return True
        
        # 2. Checagem Visual (pHash + Hamming)
        p_hash = self.calcular_phash(filepath)
        if p_hash is None:
            return False 

        for hash_existente in self.hashes_perceptuais:
            distancia_hamming = p_hash - hash_existente
            if distancia_hamming <= THRESHOLD_HAMMING:
                return True 
        
        # Se passou nos filtros, adicionamos ao banco de conhecimento
        self.hashes_sha256.add(sha_hash)
        self.hashes_perceptuais.append(p_hash)
        return False

class RoteadorSemantico:
    """Classifica a imagem usando o YOLO customizado para separar Lixo, Limpos e Oclusões."""
    def __init__(self, model_path='best.pt'):
        try:
            self.model = YOLO(model_path)
            print(f"[INFO] Modelo {model_path} carregado com sucesso.")
            # Isso vai te mostrar QUAIS classes seu modelo realmente conhece!
            print(f"[INFO] Classes conhecidas pelo seu YOLO: {self.model.names}")
        except Exception as e:
            print(f"[AVISO] Não foi possível carregar o modelo YOLO: {e}")
            self.model = None

    def classificar(self, filepath):
        if self.model is None:
            return "limpos" # Fallback de segurança: se a IA falhar, não jogue fora, passe para frente.

        resultados = self.model(filepath, verbose=False)
        result = resultados[0]
        
        # Variáveis para rastrear o que foi encontrado na imagem
        encontrou_limpo = False
        encontrou_oclusao = False
        
        # 1. Se for um modelo de Detecção de Objetos (Boxes) - O MAIS COMUM PARA ROSTOS
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                confianca = box.conf.item()
                # Flexibilizamos a confiança para 25% para evitar perder rostos
                if confianca < 0.25: 
                    continue 
                
                nome_classe = result.names[int(box.cls.item())].lower()
                
                # [DEBUG VISUAL PARA VOCÊ ENTENDER O QUE A IA ESTÁ LENDO]
                print(f"   └─ IA viu: '{nome_classe}' com {confianca*100:.1f}% de certeza")

                # Regras de roteamento flexíveis (adicionamos sinônimos)
                if any(palavra in nome_classe for palavra in ["mascara", "barba", "oclu", "sunglasses"]):
                    encontrou_oclusao = True
                elif any(palavra in nome_classe for palavra in ["face", "rosto", "limp", "person", "head"]):
                    encontrou_limpo = True

        # 2. Se for um modelo de Classificação de Imagem inteira (Probs)
        elif result.probs is not None:
            confianca = result.probs.top1conf.item()
            if confianca >= 0.25:
                nome_classe = result.names[result.probs.top1].lower()
                print(f"   └─ IA classificou foto inteira como: '{nome_classe}' ({confianca*100:.1f}%)")
                
                if any(p in nome_classe for p in ["mascara", "barba", "oclu"]):
                    encontrou_oclusao = True
                elif any(p in nome_classe for p in ["face", "rosto", "limp", "person"]):
                    encontrou_limpo = True

        # 3. DECISÃO FINAL (A Ordem de Prioridade importa)
        if encontrou_limpo:
            return "limpos"
        elif encontrou_oclusao:
            return "oclusao"
        else:
            return "lixo"

# ==========================================
# ORQUESTRADOR PRINCIPAL
# ==========================================

def configurar_ambiente():
    """Cria as pastas limpas para o run atual."""
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

def executar_pipeline():
    print("="*50)
    print("INICIANDO ID-HUNTER: MÓDULO DE TRIAGEM E DEDUPLICAÇÃO")
    print("="*50)
    
    configurar_ambiente()
    deduplicador = Deduplicador()
    
    # Apontando diretamente para o seu modelo treinado
    roteador = RoteadorSemantico(model_path='best.pt') 

    if not os.path.exists(INPUT_DIR):
        print(f"[ERRO] A pasta '{INPUT_DIR}' não existe. Crie a pasta e coloque imagens nela.")
        return

    arquivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[INFO] Encontradas {len(arquivos)} imagens para processamento.\n")

    contadores = {"limpos": 0, "oclusao": 0, "lixo": 0, "duplicados": 0}

    for arquivo in arquivos:
        filepath = os.path.join(INPUT_DIR, arquivo)
        
        # FASE 0: DEDUPLICAÇÃO
        if deduplicador.eh_duplicado(filepath):
            shutil.copy(filepath, os.path.join(DIRS["duplicados"], arquivo))
            contadores["duplicados"] += 1
            print(f"[DEDUPLICAÇÃO] {arquivo} -> Bloqueado (Cópia Exata ou Visual)")
            continue

        # FASE 1: TRIAGEM SEMÂNTICA
        categoria = roteador.classificar(filepath)
        destino = DIRS[categoria]
        shutil.copy(filepath, os.path.join(destino, arquivo))
        contadores[categoria] += 1
        print(f"[TRIAGEM] {arquivo} -> Roteado para: {categoria.upper()}")

    print("\n" + "="*50)
    print("RESUMO DA TRIAGEM:")
    print(f"- Rostos Limpos (Prontos para Extrator primário): {contadores['limpos']}")
    print(f"- Rostos com Oclusão (Pipeline secundário): {contadores['oclusao']}")
    print(f"- Lixo (Emojis/Paisagens) descartado: {contadores['lixo']}")
    print(f"- Duplicados identificados e bloqueados: {contadores['duplicados']}")
    print("="*50)

if __name__ == "__main__":
    executar_pipeline()