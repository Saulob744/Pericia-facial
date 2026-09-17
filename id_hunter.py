import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import warnings
import time

from database_manager import (
    obter_todas_tarefas_pendentes,
    obter_estatisticas_sistema,
    buscar_historico_completo_por_vetor,
    resetar_banco_dados
)
from worker import extrair_faces_tarefa

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(layout="wide", page_title="Facial-Científica | Central Unificada", page_icon="🔬")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    p, span, label, div { color: #cbd5e1; }
    .stMarkdown p { color: #cbd5e1 !important; font-size: 0.95rem; }
    
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { 
        color: #000000 !important; 
    }

    .pericia-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .pericia-title {
        color: #f8fafc !important;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0 0 8px 0;
        letter-spacing: -0.025em;
    }
    .pericia-subtitle {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        margin: 0;
    }
    .metric-card-clean, .metric-card-alert {
        background-color: #1e293b;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }
    .metric-card-clean { border-top: 3px solid #3b82f6; }
    .metric-card-alert { border-top: 3px solid #ef4444; }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #94a3b8 !important;
        margin: 8px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .badge-lote, .badge-reincidente, .badge-match {
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        letter-spacing: 0.025em;
    }
    .badge-lote { background-color: #3b82f6; color: #ffffff !important; }
    .badge-reincidente { background-color: #ef4444; color: #ffffff !important; }
    .badge-match { background-color: #10b981; color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def carregar_modelo_arcface(confianca_atual):
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=confianca_atual)          
        return app
    except Exception:
        return None

@st.fragment(run_every="3s")
def monitorar_progresso_fragment():
    stats = obter_estatisticas_sistema()
    fila_stats = stats["fila"]
    bio_stats = stats["biometria"]

    total_fila = sum(fila_stats.values())
    concluidas = fila_stats.get("CONCLUIDO", 0) + fila_stats.get("SEM_ROSTO", 0) + fila_stats.get("FALHA", 0)
    progresso_pct = (concluidas / total_fila) if total_fila > 0 else 0.0
    pendentes_reais = total_fila - concluidas

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    with col_s1:
        st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{pendentes_reais}</p><p class="metric-lbl">Pendentes</p></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#10b981;">{fila_stats.get("CONCLUIDO", 0)}</p><p class="metric-lbl">Faces Extraídas</p></div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#f59e0b;">{fila_stats.get("SEM_ROSTO", 0)}</p><p class="metric-lbl">Sem Face</p></div>', unsafe_allow_html=True)
    with col_s4:
        st.markdown(f'<div class="metric-card-alert"><p class="metric-val" style="color:#ef4444;">{bio_stats.get("total_individuos", 0)}</p><p class="metric-lbl">Cadastros Únicos</p></div>', unsafe_allow_html=True)
    with col_s5:
        st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{bio_stats.get("total_ocorrencias", 0)}</p><p class="metric-lbl">Total Ocorrências</p></div>', unsafe_allow_html=True)

    st.write("")
    st.progress(progresso_pct, text=f"Progresso de Processamento: {progresso_pct*100:.1f}% ({concluidas}/{total_fila})")
    
    if pendentes_reais <= 0 and total_fila > 0:
        st.success("🎉 Todas as tarefas da fila foram concluídas pelos workers!")

def carregar_painel():
    st.markdown("""
        <div class="pericia-header">
            <h1 class="pericia-title">Facial-Científica | Painel Unificado de Auditoria</h1>
            <p class="pericia-subtitle">Sistema de varredura automatizada, extração biométrica e busca pericial de reincidentes.</p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### ⚙️ Calibração Forense")
    confianca_minima = st.sidebar.slider("Confiança Mínima (Filtro IA)", 0.15, 0.80, 0.35, 0.05)
    tolerancia_match = st.sidebar.slider("Tolerância de Match", 0.35, 0.65, 0.45, 0.01)
    tamanho_min_face = st.sidebar.slider("Tamanho Mínimo (px)", 15, 100, 30, 5)
    st.sidebar.markdown("---")

    caminho_padrao_origem = os.getenv("PASTA_ORIGEM", r"C:\Users\saulo.borges\Downloads\facial_teste\entradas_celular")
    caminho_padrao_destino = os.getenv("PASTA_DESTINO", r"C:\Users\saulo.borges\Downloads\facial_teste\armazenamento_crops")

    aba1, aba2, aba3 = st.tabs([
        "🔬 Processamento e Fila",
        "📂 Auditoria e Reincidência",
        "🔎 Busca Biométrica"
    ])

    with aba1:
        st.subheader("Processamento de Evidências")
        st.caption("Controle de ingestão de arquivos e execução de extração facial distribuída.")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pasta_origem = st.text_input("Diretório de Origem (Lotes):", value=caminho_padrao_origem, key="cfg_origem")
        with col_p2:
            pasta_destino = st.text_input("Diretório de Destino (Crops):", value=caminho_padrao_destino, key="cfg_destino")

        st.write("")

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("📋 Alimentar Fila de Processamento"):
                if not os.path.exists(pasta_origem):
                    st.error(f"Diretório '{pasta_origem}' não localizado.")
                else:
                    with st.spinner("Indexando arquivos e inserindo em lote no PostgreSQL..."):
                        from maestro_scanner import mapear_lotes_de_extracao
                        total_enfileiradas = mapear_lotes_de_extracao(pasta_origem)
                    st.success(f"Operação concluída. {total_enfileiradas} registros inseridos na fila.")

        with col_btn2:
            if st.button("▶️ Iniciar Processamento", type="primary"):
                with st.spinner("Despachando tarefas para Redis/Celery em lote..."):
                    tarefas = obter_todas_tarefas_pendentes()
                    for t in tarefas:
                        tarefa_id, id_extracao, caminho_relativo = t
                        caminho_fisico = os.path.join(pasta_origem, caminho_relativo)
                        extrair_faces_tarefa.delay(
                            tarefa_id, id_extracao, caminho_fisico, 
                            pasta_destino, tolerancia_match, tamanho_min_face
                        )
                st.success(f"{len(tarefas)} tarefas enviadas com sucesso!")

        with st.expander("⚠️ Manutenção de Banco de Dados (Reset)"):
            confirmar_reset = st.checkbox("Estou ciente e desejo prosseguir com a exclusão dos dados.", key="chk_reset")
            if st.button("Excluir Banco de Dados", type="primary", key="btn_reset_db"):
                if confirmar_reset:
                    resetar_banco_dados()
                    st.success("Banco de dados formatado com sucesso.")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        monitorar_progresso_fragment()

    with aba2:
        st.subheader("Auditoria de Identidades e Reincidências")
        pasta_crops_audit = st.text_input("Diretório de Armazenamento de Identidades:", value=caminho_padrao_destino, key="path_crops_audit")
        
        if os.path.exists(pasta_crops_audit):
            with st.spinner("Compilando e indexando registros armazenados..."):
                ids_suspeitos = [d for d in os.listdir(pasta_crops_audit) if os.path.isdir(os.path.join(pasta_crops_audit, d))]
                total_individuos = len(ids_suspeitos)
                total_crops = 0
                reincidentes_count = 0
                lotes_mapeados = set()
                dados_suspeitos = []

                for ind_id in ids_suspeitos:
                    caminho_id = os.path.join(pasta_crops_audit, ind_id)
                    if not os.path.exists(caminho_id):
                        continue
                    fotos = [f for f in os.listdir(caminho_id) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    qtd_fotos = len(fotos)
                    total_crops += qtd_fotos
                    
                    lotes_do_suspeito = set()
                    for foto in fotos:
                        if "celular_" in foto:
                            partes = foto.split("_")
                            if len(partes) > 1:
                                lotes_do_suspeito.add(partes[1])
                        else:
                            lotes_do_suspeito.add("Geral")
                    
                    lotes_mapeados.update(lotes_do_suspeito)
                    eh_reincidente = qtd_fotos > 1 or len(lotes_do_suspeito) > 1
                    if eh_reincidente:
                        reincidentes_count += 1
                        
                    dados_suspeitos.append({
                        "id": ind_id, "caminho": caminho_id, "fotos": fotos,
                        "qtd_fotos": qtd_fotos, "lotes": list(lotes_do_suspeito), "reincidente": eh_reincidente
                    })

            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            with col_kpi1: st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{total_individuos}</p><p class="metric-lbl">Total de Cadastros</p></div>', unsafe_allow_html=True)
            with col_kpi2: st.markdown(f'<div class="metric-card-alert"><p class="metric-val" style="color:#ef4444;">{reincidentes_count}</p><p class="metric-lbl">Reincidentes</p></div>', unsafe_allow_html=True)
            with col_kpi3: st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#10b981;">{total_crops}</p><p class="metric-lbl">Evidências</p></div>', unsafe_allow_html=True)
            with col_kpi4: st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#f59e0b;">{len(lotes_mapeados)}</p><p class="metric-lbl">Lotes Analisados</p></div>', unsafe_allow_html=True)

            st.write("")
            col_filtro1, col_filtro2 = st.columns([1.2, 2])
            with col_filtro1:
                filtro_modo = st.radio("Critério de Exibição:", ["Apenas Reincidentes", "Todos os Registros"], horizontal=True, key="filtro_auditoria")
            with col_filtro2:
                termo_busca = st.text_input("Filtrar por ID ou Lote:", value="", key="busca_auditoria").strip().lower()

            st.write("")
            
            dados_filtrados = []
            for item in dados_suspeitos:
                if filtro_modo == "Apenas Reincidentes" and not item["reincidente"]: continue
                if termo_busca and not (termo_busca in item["id"].lower() or any(termo_busca in l.lower() for l in item["lotes"])): continue
                dados_filtrados.append(item)
            
            LIMITE_EXIBICAO = 50
            if len(dados_filtrados) > LIMITE_EXIBICAO:
                st.info(f"Otimização de Tela: Exibindo {LIMITE_EXIBICAO} registros de {len(dados_filtrados)}. Utilize os filtros acima para refinar a busca.")

            for item in dados_filtrados[:LIMITE_EXIBICAO]:
                status_badge = '<span class="badge-reincidente">REINCIDENTE</span>' if item["reincidente"] else '<span class="badge-lote">REGISTRO ÚNICO</span>'
                with st.expander(f"Identificação: {item['id']} | {item['qtd_fotos']} evidência(s)", expanded=False):
                    st.markdown(f"{status_badge} &nbsp; <span style='color:#94a3b8; font-size:0.85rem;'><b>Origem:</b> {', '.join(item['lotes'])}</span>", unsafe_allow_html=True)
                    st.write("")
                    cols = st.columns(min(len(item["fotos"]), 7))
                    for idx, foto_nome in enumerate(item["fotos"][:14]):
                        caminho_foto = os.path.join(item["caminho"], foto_nome)
                        with cols[idx % 7]:
                            if os.path.exists(caminho_foto):
                                st.image(caminho_foto, width=100)

    with aba3:
        st.subheader("Busca de Identidades por Referência")
        c1, c2 = st.columns([1.2, 1])
        
        with c1:
            uploaded_file = st.file_uploader("Selecionar Arquivo para Busca:", type=["jpg", "jpeg", "png"], key="up_ref")
            img_bgr = None
            if uploaded_file is not None:
                imagem_bytes_upload = uploaded_file.read()
                nparr = np.frombuffer(imagem_bytes_upload, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with c2:
            if img_bgr is not None:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=180, caption="Referência")

        if st.button("Executar Pesquisa Vetorial", type="primary", key="btn_ref"):
            if img_bgr is None:
                st.error("Forneça uma imagem válida.")
            else:
                with st.spinner("Extraindo vetores biométricos e varrendo banco de dados..."):
                    app_arcface = carregar_modelo_arcface(confianca_minima)
                    if app_arcface is None:
                        st.error("Erro interno ao carregar dependências biométricas.")
                    else:
                        faces_raw = app_arcface.get(img_bgr)
                        faces_validas = [f for f in faces_raw if (min(f.bbox[2]-f.bbox[0], f.bbox[3]-f.bbox[1]) >= tamanho_min_face)]
                        
                        if len(faces_validas) == 0:
                            st.warning("Não foi possível isolar características faciais.")
                        else:
                            st.success(f"{len(faces_validas)} biometria(s) extraídas. Analisando resultados...")
                            for idx_f, face in enumerate(faces_validas, start=1):
                                vetor_512d = face.embedding.astype(np.float32)
                                resultados = buscar_historico_completo_por_vetor(vetor_512d, limite_distancia=tolerancia_match)
                                
                                st.write(f"#### Perfil #{idx_f}")
                                if not resultados:
                                    st.info("Sem correspondência histórica.")
                                else:
                                    for res in resultados:
                                        ind_id, dist, ocorrencias = res["individuo_id"], res["distancia"], res["ocorrencias"]
                                        col_res1, col_res2 = st.columns([1, 2.2])
                                        
                                        with col_res1:
                                            if dist <= 0.38: st.markdown('<span class="badge-reincidente">MATCH POSITIVO</span>', unsafe_allow_html=True)
                                            else: st.markdown('<span class="badge-lote">CORRESPONDÊNCIA PARCIAL</span>', unsafe_allow_html=True)
                                            st.markdown(f"**ID:** `{ind_id}`<br>**Proximidade:** `{dist:.4f}`", unsafe_allow_html=True)
                                        
                                        with col_res2:
                                            pasta_historico = os.path.join(caminho_padrao_destino, str(ind_id))
                                            if os.path.exists(pasta_historico):
                                                fotos_historico = [f for f in os.listdir(pasta_historico) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                                if fotos_historico:
                                                    cols_img = st.columns(min(len(fotos_historico), 5))
                                                    for i_img, f_nome in enumerate(fotos_historico[:5]):
                                                        with cols_img[i_img]:
                                                            st.image(os.path.join(pasta_historico, f_nome), width=80)
                                            
                                            df_oc = pd.DataFrame(ocorrencias)
                                            if not df_oc.empty:
                                                st.dataframe(df_oc[['lote', 'caminho_arquivo', 'data_ocorrencia']])

if __name__ == "__main__":
    carregar_painel()