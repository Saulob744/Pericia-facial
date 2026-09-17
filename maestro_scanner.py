import os
import argparse
from database_manager import enfileirar_tarefas_em_lote

def mapear_lotes_de_extracao(diretorio_alvo):
    print("=" * 60)
    print("🔍 MAESTRO: Lendo pastas e alimentando a Fila de Processamento...")
    print(f"📂 Diretório de Origem: '{diretorio_alvo}'")
    print("=" * 60)

    if not os.path.exists(diretorio_alvo):
        print(f"[ERRO FATAL] O diretório '{diretorio_alvo}' não foi encontrado no sistema.")
        return 0

    lotes = [d for d in os.listdir(diretorio_alvo) if os.path.isdir(os.path.join(diretorio_alvo, d))]

    if not lotes:
        print(f"[AVISO] Nenhuma subpasta encontrada em '{diretorio_alvo}'.")
        return 0

    total_imagens_geral = 0
    todas_as_tarefas = []

    for lote in lotes:
        caminho_lote = os.path.join(diretorio_alvo, lote)
        imagens_enfileiradas_neste_lote = 0
        
        for root, _, files in os.walk(caminho_lote):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    caminho_absoluto = os.path.join(root, file)
                    caminho_relativo = os.path.relpath(caminho_absoluto, diretorio_alvo)
                    
                    todas_as_tarefas.append({
                        "id_extracao": lote,
                        "caminho_arquivo": caminho_relativo
                    })
                    
                    imagens_enfileiradas_neste_lote += 1
                    total_imagens_geral += 1

        print(f"📁 Lote '{lote}': {imagens_enfileiradas_neste_lote} tarefas mapeadas.")

    if todas_as_tarefas:
        enfileirar_tarefas_em_lote(todas_as_tarefas)

    print("-" * 60)
    print(f"🎯 RESUMO: {len(lotes)} Celulares/Lotes mapeados.")
    print(f"🎯 TOTAL: {total_imagens_geral} tarefas inseridas em lote no banco de dados.")
    print("=" * 60)
    
    return total_imagens_geral

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scanner de Pastas Periciais")
    parser.add_argument(
        "--origem", 
        type=str, 
        required=True, 
        help="Caminho completo da pasta onde estão os lotes (ex: C:\\Pericia\\Fotos)"
    )
    
    args = parser.parse_args()
    mapear_lotes_de_extracao(args.origem)