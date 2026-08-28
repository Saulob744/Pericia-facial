import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

# ==========================================
# CONFIGURAÇÕES DO BANCO
# ==========================================
DB_CONFIG = {
    "dbname": "idhunter_db",
    "user": "perito",
    "password": "senha_segura_123",
    "host": "127.0.0.1",
    "port": "5454"
}

def conectar_banco():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def inicializar_banco():
    conn = conectar_banco()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS individuos (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            embedding VECTOR(512),
            data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ocorrencias (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            individuo_id UUID REFERENCES individuos(id),
            id_extracao VARCHAR(50), 
            hash_sha256 VARCHAR(64) UNIQUE,
            caminho_arquivo TEXT,
            caminho_crop TEXT,
            data_ocorrencia TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fila_processamento (
            id SERIAL PRIMARY KEY,
            id_extracao VARCHAR(50),
            caminho_arquivo TEXT UNIQUE,
            status VARCHAR(20) DEFAULT 'PENDENTE',
            data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("[BANCO] Tabelas 'individuos', 'ocorrencias' e 'fila_processamento' estruturadas.")

def buscar_similar(vetor_novo, limite_distancia=0.45):
    conn = conectar_banco()
    cursor = conn.cursor()
    
    sql = """
        SELECT id, (embedding <=> %s) AS distancia 
        FROM individuos 
        WHERE (embedding <=> %s) < %s
        ORDER BY distancia ASC
        LIMIT 1;
    """
    cursor.execute(sql, (vetor_novo, vetor_novo, limite_distancia))
    resultado = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return resultado

def registrar_face(vetor_512d, hash_foto, id_extracao, caminho_original, caminho_crop):
    conn = conectar_banco()
    cursor = conn.cursor()
    historico_lotes = []
    
    try:
        match = buscar_similar(vetor_512d)
        
        if match:
            individuo_id = match[0]
            status = "REINCIDENTE"
            
            cursor.execute("""
                SELECT DISTINCT id_extracao 
                FROM ocorrencias 
                WHERE individuo_id = %s;
            """, (individuo_id,))
            
            historico_lotes = [linha[0] for linha in cursor.fetchall()]
            
        else:
            sql_novo = "INSERT INTO individuos (embedding) VALUES (%s) RETURNING id;"
            cursor.execute(sql_novo, (vetor_512d,))
            individuo_id = cursor.fetchone()[0]
            status = "NOVO"
            
        sql_ocorrencia = """
            INSERT INTO ocorrencias (individuo_id, id_extracao, hash_sha256, caminho_arquivo, caminho_crop)
            VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(sql_ocorrencia, (individuo_id, id_extracao, hash_foto, caminho_original, caminho_crop))
        
        conn.commit()
        
        return status, individuo_id, historico_lotes
        
    except psycopg2.IntegrityError:
        conn.rollback()
        return "DUPLICADO", None, []
    finally:
        cursor.close()
        conn.close()

def enfileirar_tarefa(id_extracao, caminho_arquivo):
    conn = conectar_banco()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO fila_processamento (id_extracao, caminho_arquivo, status)
            VALUES (%s, %s, 'PENDENTE')
            ON CONFLICT (caminho_arquivo) DO NOTHING;
        """
        cursor.execute(sql, (id_extracao, caminho_arquivo))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    inicializar_banco()

def pegar_tarefa_pendente():
    conn = conectar_banco()
    cursor = conn.cursor()
    try:
        sql = """
            UPDATE fila_processamento
            SET status = 'PROCESSANDO'
            WHERE id = (
                SELECT id FROM fila_processamento 
                WHERE status = 'PENDENTE' 
                ORDER BY id ASC
                LIMIT 1 
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, id_extracao, caminho_arquivo;
        """
        cursor.execute(sql)
        tarefa = cursor.fetchone()
        conn.commit()
        return tarefa
    except Exception as e:
        conn.rollback()
        print(f"[ERRO BANCO] Falha ao puxar tarefa: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def atualizar_status_tarefa(tarefa_id, novo_status):
    conn = conectar_banco()
    cursor = conn.cursor()
    try:
        sql = "UPDATE fila_processamento SET status = %s WHERE id = %s;"
        cursor.execute(sql, (novo_status, tarefa_id))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def obter_estatisticas_sistema():
    conn = conectar_banco()
    cursor = conn.cursor()
    
    estatisticas = {
        "fila": {"PENDENTE": 0, "PROCESSANDO": 0, "CONCLUIDO": 0, "FALHA": 0, "SEM_ROSTO": 0},
        "biometria": {"total_individuos": 0, "total_ocorrencias": 0}
    }
    
    try:
        cursor.execute("SELECT status, COUNT(*) FROM fila_processamento GROUP BY status;")
        for status, quantidade in cursor.fetchall():
            estatisticas["fila"][status] = quantidade
            
        cursor.execute("SELECT COUNT(*) FROM individuos;")
        estatisticas["biometria"]["total_individuos"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ocorrencias;")
        estatisticas["biometria"]["total_ocorrencias"] = cursor.fetchone()[0]
        
        return estatisticas
    finally:
        cursor.close()
        conn.close()