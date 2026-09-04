import os
import cv2
import argparse
import numpy as np
import warnings
from insightface.app import FaceAnalysis
from database_manager import buscar_historico_completo_por_vetor

warnings.filterwarnings('ignore', category=FutureWarning)

PATH_PADRAO_REFERENCIA = r"C:\Users\saulo.borges\Downloads\facial_teste\referencia"

def desenhar_barra_progresso(atual, total, tamanho=30):
    porcentagem = float(atual) / total
    blocos = int(round(tamanho * porcentagem))
    barra = "█" * blocos + "░" * (tamanho - blocos)
    print(f"\r⏳ [PROCESSANDO] [{barra}] {atual}/{total} ({porcentagem*100:.1f}%)", end="", flush=True)

def processar_busca_referencia(caminho_referencia: str, limite_distancia: float = 0.38):
    print("=" * 75)
    print(" 🎯 ID HUNTER - BUSCA PERICIAL POR FOTO DE REFERÊNCIA")
    print("=" * 75)
    
    if not caminho_referencia:
        caminho_referencia = PATH_PADRAO_REFERENCIA
        
    print(f"📂 Alvo de Referência: '{caminho_referencia}'")
    
    if not os.path.exists(caminho_referencia):
        print(f"\n❌ [ERRO] O caminho '{caminho_referencia}' não foi encontrado no sistema.")
        return

    arquivos_referencia = []
    if os.path.isfile(caminho_referencia):
        if caminho_referencia.lower().endswith(('.png', '.jpg', '.jpeg')):
            arquivos_referencia.append(caminho_referencia)
    elif os.path.isdir(caminho_referencia):
        for f in os.listdir(caminho_referencia):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                arquivos_referencia.append(os.path.join(caminho_referencia, f))

    if not arquivos_referencia:
        print(f"\n⚠️ [AVISO] Nenhuma imagem (.png, .jpg, .jpeg) encontrada em '{caminho_referencia}'.")
        return

    print(f"🖼️ Encontrada(s) {len(arquivos_referencia)} foto(s) de referência para análise.")
    print("🧠 Carregando motor biométrico Híbrido HD (InsightFace 1024px + OpenCV Fallback)...")
    
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.25)
    
    cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    print("[OK] Motor biométrico de alta sensibilidade pronto.\n")

    total_arquivos = len(arquivos_referencia)

    for arq_idx, caminho_foto in enumerate(arquivos_referencia, start=1):
        nome_arquivo = os.path.basename(caminho_foto)
        print("\n" + "─" * 75)
        desenhar_barra_progresso(arq_idx, total_arquivos)
        print(f"\n📸 [{arq_idx}/{total_arquivos}] Analisando: '{nome_arquivo}'")
        
        img = cv2.imread(caminho_foto)
        if img is None:
            print("   └─ ❌ [ERRO] Não foi possível ler a imagem.")
            continue
            
        faces = app.get(img)
        if len(faces) == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(rects) == 0:
                rects = cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            if len(rects) > 0:
                print(f"   ├─ 🛡️ [REDE DE SEGURANÇA OPENCV] Recuperou {len(rects)} rosto(s) na referência!")
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
            print("   └─ 👻 [IGNORADO] Nenhum rosto encontrado na referência.")
            continue
            
        faces_validas = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
            altura = y2 - y1
            largura = x2 - x1
            proporcao = largura / float(altura) if altura > 0 else 0
            
            if altura >= 30 and largura >= 30 and 0.35 <= proporcao <= 2.5:
                faces_validas.append(face)

        if not faces_validas:

            print("   └─ 🗑️ [IGNORADO] Rosto muito pequeno ou proporção inválida (ruído/cabelo).")
            continue

        print(f"   ├─ 🧠 Encontrado(s) {len(faces_validas)} rosto(s) válido(s) na imagem.")
        
        for f_idx, face in enumerate(faces_validas, start=1):
            vetor_512d = face.embedding.astype(np.float32)
            
            # Consulta com limite de 0.42 (pega 0.38 estrito e ate 0.42 como alerta)
            resultados = buscar_historico_completo_por_vetor(vetor_512d, limite_distancia=0.42)
            
            print(f"\n   👤 [Rosto {f_idx}/{len(faces_validas)}]")
            
            if not resultados:
                print("   │  ℹ️ INDIVÍDUO INÉDITO: Nenhuma ocorrência ou match encontrado no banco de dados.")
            else:
                for res in resultados:
                    ind_id = res["individuo_id"]
                    dist = res["distancia"]
                    ocorrencias = res["ocorrencias"]
                    
                    lotes_unicos = sorted(list(set(oc["lote"] for oc in ocorrencias)))
                    
                    if dist <= limite_distancia:
                        print(f"   │  🚨 MATCH CONFIRMADO (Distância: {dist:.4f})")
                    else:
                        print(f"   │  ⚠️ BIOMETRIA PRÓXIMA / SUSPEITO (Distância: {dist:.4f})")
                        
                    print(f"   │     ├─ 🆔 ID do Indivíduo no Banco: {ind_id}")
                    print(f"   │     ├─ 📊 Total de Aparições: {len(ocorrencias)} foto(s) em {len(lotes_unicos)} celular(es)/lote(s)")
                    print(f"   │     ├─ 📱 REDE DE CELULARES ONDE APARECEU:")
                    for lote in lotes_unicos:
                        count_lote = sum(1 for oc in ocorrencias if oc["lote"] == lote)
                        print(f"   │     │    ├─ 📁 Lote/Celular: '{lote}' ({count_lote} aparição/ões)")
                    
                    print(f"   │     └─ 📸 HISTÓRICO DE OCORRÊNCIAS:")
                    for oc_idx, oc in enumerate(ocorrencias[:10], start=1):
                        data_str = oc["data_ocorrencia"].strftime("%d/%m/%Y %H:%M") if oc["data_ocorrencia"] is not None else "N/A"
                        print(f"   │          ├─ [{oc_idx}] Celular: {oc['lote']} | Data: {data_str}")
                        print(f"   │          │    └─ Foto Original: {oc['caminho_arquivo']}")
                    if len(ocorrencias) > 10:
                        print(f"   │          └─ ... e mais {len(ocorrencias) - 10} ocorrência(s).")
                    print("   │")

    print("\n" + "=" * 75)
    print("✅ BUSCA PERICIAL DE REFERÊNCIA FINALIZADA")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busca de Indivíduo por Foto de Referência")
    parser.add_argument("--referencia", type=str, default=PATH_PADRAO_REFERENCIA, help="Caminho da imagem ou pasta de referência")
    args = parser.parse_args()
    
    processar_busca_referencia(args.referencia)
