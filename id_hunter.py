import os
import sys
import time
import subprocess

PASTA_ORIGEM = r"C:\Users\saulo.borges\Downloads\facial_teste\entradas_celular"
PASTA_DESTINO = r"C:\Users\saulo.borges\Downloads\facial_teste\armazenamento_crops"

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def obter_pastas():
    global PASTA_ORIGEM, PASTA_DESTINO
    
    padrao_origem = r"C:\Users\saulo.borges\Downloads\facial_teste\entradas_celular"
    padrao_destino = r"C:\Users\saulo.borges\Downloads\facial_teste\armazenamento_crops"
    
    if not PASTA_ORIGEM:
        print(f"\n📂 [ENTRADA] Pressione ENTER para usar a pasta padrão:\n    '{padrao_origem}'")
        resp = input("Ou digite um novo caminho de Origem: ").strip()
        PASTA_ORIGEM = resp if resp else padrao_origem
    
    if not PASTA_DESTINO:
        print(f"\n💾 [SAÍDA] Pressione ENTER para usar a pasta padrão:\n    '{padrao_destino}'")
        resp = input("Ou digite um novo caminho de Destino: ").strip()
        PASTA_DESTINO = resp if resp else padrao_destino
        
    os.makedirs(PASTA_DESTINO, exist_ok=True)
    return PASTA_ORIGEM, PASTA_DESTINO


def executar_com_debug(script_e_argumentos):
    comando_completo = [sys.executable] + script_e_argumentos
    print(f"\n🔧 [SISTEMA] Executando comando: {' '.join(comando_completo)}\n")
    
    try:
        resultado = subprocess.run(comando_completo)
        
        if resultado.returncode != 0:
            print("\n" + "!" * 70)
            print(f"❌ [ERRO FATAL] O script parou de funcionar (Código {resultado.returncode}).")
            print("LEIA O ERRO ACIMA ANTES DE CONTINUAR.")
            print("!" * 70)
            input("\nPressione ENTER para voltar ao menu...")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário.")
    except Exception as e:
        print(f"\n❌ [ERRO DE SISTEMA] Não consegui nem iniciar o comando: {e}")
        input("\nPressione ENTER para voltar ao menu...")

def exibir_menu():
    limpar_tela()
    print("=" * 70)
    print(" 🕵️‍♂️ ID HUNTER - CENTRAL DE COMANDO DA PERÍCIA ")
    print("=" * 70)
    print(f" 📂 Origem (Evidências): {PASTA_ORIGEM if PASTA_ORIGEM else 'Nenhuma definida'}")
    print(f" 💾 Destino (Crops):     {PASTA_DESTINO if PASTA_DESTINO else 'Nenhum definido'}")
    print("-" * 70)
    print(" 1. 🔍 Escanear Celulares (Adicionar na Fila)")
    print(" 2. 👷 Iniciar Operário de IA (Processar Fila)")
    print(" 3. 📊 Abrir Painel de Monitoramento ao Vivo")
    print(" 6. 🎯 Buscar Indivíduo por Foto de Referência (Mapear Rede de Lotes)")
    print(" 4. 🔄 Mudar Pastas (Origem/Destino)")
    print(" 5. 🧹 RESETAR SISTEMA (Limpar Banco de Dados)")
    print(" 0. 🚪 Sair do Sistema")
    print("=" * 70)
    return input("Escolha uma opção: ")

def main():
    global PASTA_ORIGEM, PASTA_DESTINO
    
    while True:
        opcao = exibir_menu()
        
        if opcao == '1':
            origem, _ = obter_pastas()
            limpar_tela()
            executar_com_debug(['maestro_scanner.py', '--origem', origem])
            input("\n[Pressione ENTER para voltar ao menu principal...]")
            
        elif opcao == '2':
            origem, destino = obter_pastas()
            limpar_tela()
            print("👷 Iniciando Operário de Inteligência Artificial...")
            print("⚠️  (Para parar o operário e voltar ao menu, pressione CTRL+C)\n")
            time.sleep(2)
            executar_com_debug(['worker_operario.py', '--origem', origem, '--destino', destino])
                
        elif opcao == '3':
            limpar_tela()
            print("🌐 Abrindo Painel Web no navegador (Streamlit)...")
            executar_com_debug(['-m', 'streamlit', 'run', 'painel_web.py'])


        elif opcao == '6':
            limpar_tela()
            path_padrao = r"C:\Users\saulo.borges\Downloads\facial_teste\referencia"
            print("=" * 70)
            print(" 🎯 BUSCA PERICIAL POR FOTO DE REFERÊNCIA ")
            print("=" * 70)
            print(f"Pressione ENTER para usar a pasta padrão: '{path_padrao}'")
            entrada_ref = input("Ou digite o caminho da foto/pasta de referência: ").strip()
            caminho_final = entrada_ref if entrada_ref else path_padrao
            
            executar_com_debug(['buscar_referencia.py', '--referencia', caminho_final])
            input("\n[Pressione ENTER para voltar ao menu principal...]")
                
        elif opcao == '4':
            PASTA_ORIGEM = input("\nDigite o novo caminho de ORIGEM: ").strip()
            PASTA_DESTINO = input("Digite o novo caminho de DESTINO: ").strip()
            os.makedirs(PASTA_DESTINO, exist_ok=True)
            print("Pastas atualizadas com sucesso!")
            time.sleep(1)
            
        elif opcao == '5':
            print("\n⚠️ AVISO: Isso apagará todas as ocorrências e rostos do Banco de Dados!")
            if input("Tem certeza absoluta? (S/N): ").strip().lower() == 's':
                print("Limpando banco...")
                comando = ['docker', 'exec', '-it', 'idhunter_postgres', 'psql', '-U', 'perito', '-d', 'idhunter_db', '-c', 'TRUNCATE fila_processamento, ocorrencias, individuos RESTART IDENTITY CASCADE;']
                subprocess.run(comando)
                print("✅ Banco zerado com sucesso!")
            input("\n[Pressione ENTER para voltar ao menu principal...]")
            
        elif opcao == '0':
            sys.exit(0)
        else:
            print("\n❌ Opção inválida.")
            time.sleep(1)


if __name__ == "__main__":
    main()