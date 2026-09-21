import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "idhunter_db"),
    "user": os.getenv("POSTGRES_USER", "perito"),
    "password": os.getenv("POSTGRES_PASSWORD", "senha_segura_123"),
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": os.getenv("DB_PORT", "5454")
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
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_individuos_embedding_hnsw 
        ON individuos 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
    """)
    conn.commit()
    cursor.close()
    conn.close()

def resetar_banco_dados():
    conn = conectar_banco()
    cursor = conn.cursor()
    cursor.execute("TRUNCATE fila_processamento, ocorrencias, individuos RESTART IDENTITY CASCADE;")
    conn.commit()
    cursor.close()
    conn.close()
    return True

def buscar_similar(vetor_novo, limite_distancia=0.50):
    conn = conectar_banco()
    cursor = conn.cursor()
    vetor_arr = np.array(vetor_novo, dtype=np.float32)
    norma = np.linalg.norm(vetor_arr)
    if norma > 0:
        vetor_arr = vetor_arr / norma
    sql = """
        SELECT id, (embedding <=> %s::vector) AS distancia 
        FROM individuos 
        WHERE (embedding <=> %s::vector) < %s
        ORDER BY distancia ASC
        LIMIT 1;
    """
    cursor.execute(sql, (vetor_arr, vetor_arr, limite_distancia))
    resultado = cursor.fetchone()
    cursor.close()
    conn.close()
    return resultado

def buscar_historico_completo_por_vetor(vetor_novo, limite_distancia=0.50):
    conn = conectar_banco()
    cursor = conn.cursor()
    vetor_arr = np.array(vetor_novo, dtype=np.float32)
    norma = np.linalg.norm(vetor_arr)
    if norma > 0:
        vetor_arr = vetor_arr / norma
    sql_ind = """
        SELECT id, (embedding <=> %s::vector) AS distancia 
        FROM individuos 
        WHERE (embedding <=> %s::vector) < %s
        ORDER BY distancia ASC;
    """
    cursor.execute(sql_ind, (vetor_arr, vetor_arr, limite_distancia))
    individuos_encontrados = cursor.fetchall()
    resultados = []
    for ind_id, dist in individuos_encontrados:
        sql_ocorrencias = """
            SELECT id_extracao, caminho_arquivo, caminho_crop, data_ocorrencia
            FROM ocorrencias
            WHERE individuo_id = %s
            ORDER BY data_ocorrencia DESC;
        """
        cursor.execute(sql_ocorrencias, (ind_id,))
        ocorrencias = cursor.fetchall()
        resultados.append({
            "individuo_id": ind_id,
            "distancia": float(dist),
            "ocorrencias": [
                {
                    "lote": oc[0],
                    "caminho_arquivo": oc[1],
                    "caminho_crop": oc[2],
                    "data_ocorrencia": oc[3]
                }
                for oc in ocorrencias
            ]
        })
    cursor.close()
    conn.close()
    return resultados

def registrar_face(vetor_512d, hash_foto, id_extracao, caminho_original, caminho_crop, limite_distancia=0.50):
    conn = conectar_banco()
    cursor = conn.cursor()
    historico_lotes = []
    vetor_arr = np.array(vetor_512d, dtype=np.float32)
    norma = np.linalg.norm(vetor_arr)
    if norma > 0:
        vetor_arr = vetor_arr / norma
    try:
        match = buscar_similar(vetor_arr, limite_distancia=limite_distancia)
        if match:
            individuo_id = match[0]
            status = "REINCIDENTE"
            try:
                cursor.execute("SELECT embedding FROM individuos WHERE id = %s;", (individuo_id,))
                row_emb = cursor.fetchone()
                if row_emb and row_emb[0] is not None:
                    v_old = np.array(row_emb[0], dtype=np.float32)
                    v_blend = 0.80 * v_old + 0.20 * vetor_arr
                    norm_blend = np.linalg.norm(v_blend)
                    if norm_blend > 0:
                        v_blend = v_blend / norm_blend
                    cursor.execute("UPDATE individuos SET embedding = %s WHERE id = %s;", (v_blend, individuo_id))
            except Exception:
                pass
            cursor.execute("SELECT DISTINCT id_extracao FROM ocorrencias WHERE individuo_id = %s;", (individuo_id,))
            historico_lotes = [linha[0] for linha in cursor.fetchall()]
        else:
            sql_novo = "INSERT INTO individuos (embedding) VALUES (%s) RETURNING id;"
            cursor.execute(sql_novo, (vetor_arr,))
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

def enfileirar_tarefas_em_lote(lista_tarefas):
    if not lista_tarefas:
        return 0
    conn = conectar_banco()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO fila_processamento (id_extracao, caminho_arquivo)
            VALUES %s
            ON CONFLICT (caminho_arquivo) DO NOTHING;
        """
        valores = [(t["id_extracao"], t["caminho_arquivo"]) for t in lista_tarefas]
        execute_values(cursor, sql, valores)
        conn.commit()
        return len(valores)
    finally:
        cursor.close()
        conn.close()

def obter_todas_tarefas_pendentes():
    conn = conectar_banco()
    cursor = conn.cursor()
    try:
        sql = "SELECT id, id_extracao, caminho_arquivo FROM fila_processamento WHERE status = 'PENDENTE';"
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

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
    except Exception:
        conn.rollback()
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

def cancelar_tarefas_pendentes():
    conn = conectar_banco()
    cursor = conn.cursor()
    try:
        sql = "UPDATE fila_processamento SET status = 'CANCELADO' WHERE status IN ('PENDENTE', 'PROCESSANDO');"
        cursor.execute(sql)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    inicializar_banco()