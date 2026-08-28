import os
import time
from database_manager import obter_estatisticas_sistema

def limpar_tela():
    """Limpa o terminal (compatível com Windows e Linux)"""
    os.system('cls' if os.name == 'nt' else 'clear')

def iniciar_monitoramento():
    while True:
        stats = obter_estatisticas_sistema()
        fila = stats["fila"]
        bio = stats["biometria"]
        
        # Calcula o total de fotos já catalogadas na fila
        total_fotos = sum(fila.values())
        concluidas = fila.get("CONCLUIDO", 0) + fila.get("SEM_ROSTO", 0) + fila.get("FALHA", 0)
        
        # Calcula a porcentagem de progresso
        progresso = (concluidas / total_fotos * 100) if total_fotos > 0 else 0

        limpar_tela()
        print("=" * 60)
        print(" 📡 TORRE DE CONTROLE: ID HUNTER - STATUS AO VIVO")
        print("=" * 60)
        
        print("\n📦 [STATUS DA FILA DE IMAGENS]")
        print(f"   ├─ 🕒 Pendentes (Aguardando): {fila.get('PENDENTE', 0)}")
        print(f"   ├─ ⚙️  Processando Agora:    {fila.get('PROCESSANDO', 0)} (Workers Ativos)")
        print(f"   ├─ ✅ Sucesso (Com Rosto):  {fila.get('CONCLUIDO', 0)}")
        print(f"   ├─ 👻 Sem Rosto Detectado:  {fila.get('SEM_ROSTO', 0)}")
        print(f"   └─ ❌ Falhas (Corrompidas): {fila.get('FALHA', 0)}")
        print(f"\n   📊 Progresso Geral: {progresso:.1f}% de {total_fotos} imagens processadas.")
        
        print("\n" + "-" * 60)
        print("👤 [INTELIGÊNCIA BIOMÉTRICA]")
        print(f"   ├─ 🧬 Indivíduos Únicos Cadastrados: {bio['total_individuos']}")
        print(f"   └─ 📍 Ocorrências (Aparições em Lotes): {bio['total_ocorrencias']}")
        print("=" * 60)
        print("Atualizando a cada 3 segundos... (Pressione CTRL+C para sair)")
        
        time.sleep(3)

if __name__ == "__main__":
    iniciar_monitoramento()