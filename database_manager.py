import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

# Configurações do Banco (As mesmas do docker-compose.yml)
DB_CONFIG = {
    "dbname": "idhunter_db",
    "user": "perito",
    "password": "senha_segura_123",
    "host": "127.0.0.1",
    "port": "5454"
}

def conectar_banco():
    """Cria a conexão com o PostgreSQL e ensina ele a ler vetores."""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn) # Ensina o psycopg2 a lidar com o pgvector
    return conn

def inserir_face(hash_foto, caminho_original, caminho_crop, vetor_512d):
    """Salva a face recortada e a matemática dela no banco."""
    conn = conectar_banco()
    cursor = conn.cursor()
    
    try:
        sql = """
            INSERT INTO ppl_faces (hash_sha256, caminho_origem_nas, path_minio_crop, embedding)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """
        cursor.execute(sql, (hash_foto, caminho_original, caminho_crop, vetor_512d))
        novo_id = cursor.fetchone()[0]
        conn.commit()
        print(f"[OK] Rosto salvo com sucesso! ID no Banco: {novo_id}")
        return novo_id
    except psycopg2.IntegrityError:
        print("[AVISO] Este Hash SHA-256 já existe no banco. Inserção bloqueada.")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def buscar_similar(vetor_novo, limite_distancia=0.5):
    """Busca no banco rostos parecidos usando a mágica do HNSW."""
    conn = conectar_banco()
    cursor = conn.cursor()
    
    # O operador <=> faz a busca por similaridade de cosseno (a melhor para rostos)
    sql = """
        SELECT id, caminho_origem_nas, (embedding <=> %s) AS distancia 
        FROM ppl_faces 
        WHERE (embedding <=> %s) < %s
        ORDER BY distancia ASC
        LIMIT 1;
    """
    cursor.execute(sql, (vetor_novo, vetor_novo, limite_distancia))
    resultado = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return resultado

# ==========================================
# ÁREA DE TESTE
# ==========================================
if __name__ == "__main__":
    print("Iniciando teste de conexão com o banco corporativo...")
    
    # 1. Simular a Extração de uma Face (Geramos 512 números aleatórios)
    meu_vetor_teste = np.random.rand(512).astype(np.float32)
    hash_falso = "abc123hashfalso_teste_001"
    
    print("\n--- TESTE 1: INSERÇÃO ---")
    inserir_face(
        hash_foto=hash_falso,
        caminho_original="Celular_Suspeito/foto_praia.jpg",
        caminho_crop="minio_bucket/rosto_01.jpg",
        vetor_512d=meu_vetor_teste
    )
    
    print("\n--- TESTE 2: BUSCA (O Rosto acabou de chegar de novo) ---")
    # Vamos buscar o próprio vetor para ver se o banco acha ele mesmo
    match = buscar_similar(meu_vetor_teste)
    
    if match:
        print(f"🚨 ALERTA: Sujeito encontrado!")
        print(f"ID: {match[0]}")
        print(f"Pasta Original: {match[1]}")
        print(f"Distância (0 é idêntico): {match[2]:.4f}")
    else:
        print("Nenhum suspeito encontrado no banco.")