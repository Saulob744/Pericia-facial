import streamlit as st
import os
import cv2
import numpy as np
import warnings

from database_manager import (
    obter_todas_tarefas_pendentes,
    obter_estatisticas_sistema,
    buscar_historico_completo_por_vetor,
    resetar_banco_dados,
    pegar_tarefa_pendente
)

from worker import (
    extrair_faces_tarefa
)

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(layout="wide", page_title="PERÍCIA FACE | Polícia Científica", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stApp { background-color: #0b0f19; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #0f172a !important; }
    .pericia-header { display: flex; align-items: center; justify-content: space-between; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #2563eb; margin-bottom: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.4); }
    .pericia-title { color: #f8fafc !important; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .pericia-subtitle { color: #94a3b8 !important; font-size: 1.1rem; margin: 0; }
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); border: 1px solid #334155; }
    .metric-val { font-size: 2rem; font-weight: 800; margin: 0; line-height: 1; }
    .metric-lbl { font-size: 0.85rem; color: #cbd5e1 !important; margin-top: 10px; text-transform: uppercase; font-weight: 600; }
    .badge { padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .badge-reincidente { background-color: #dc2626; color: #ffffff; }
    .result-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #3b82f6; margin-bottom: 15px; }
    .logo-container { display: flex; justify-content: flex-end; align-items: center; }
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
    fila_stats = stats.get("fila", {})
    bio_stats = stats.get("biometria", {})
    total_fila = sum(fila_stats.values())
    concluidas = fila_stats.get("CONCLUIDO", 0) + fila_stats.get("SEM_ROSTO", 0) + fila_stats.get("FALHA", 0)
    progresso_pct = (concluidas / total_fila) if total_fila > 0 else 0.0
    pendentes_reais = total_fila - concluidas

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown(f'<div class="metric-card" style="border-top: 4px solid #3b82f6;"><p class="metric-val" style="color:#60a5fa;">{pendentes_reais}</p><p class="metric-lbl">Pendentes</p></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card" style="border-top: 4px solid #10b981;"><p class="metric-val" style="color:#34d399;">{fila_stats.get("CONCLUIDO", 0)}</p><p class="metric-lbl">Faces Extraídas</p></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card" style="border-top: 4px solid #f59e0b;"><p class="metric-val" style="color:#fbbf24;">{fila_stats.get("SEM_ROSTO", 0)}</p><p class="metric-lbl">Sem Face</p></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card" style="border-top: 4px solid #ef4444;"><p class="metric-val" style="color:#f87171;">{bio_stats.get("total_individuos", 0)}</p><p class="metric-lbl">Cadastros Únicos</p></div>', unsafe_allow_html=True)
    with col5: st.markdown(f'<div class="metric-card" style="border-top: 4px solid #8b5cf6;"><p class="metric-val" style="color:#a78bfa;">{bio_stats.get("total_ocorrencias", 0)}</p><p class="metric-lbl">Total Ocorrências</p></div>', unsafe_allow_html=True)

    st.write("")
    st.progress(progresso_pct, text=f"Progresso de Processamento: {progresso_pct*100:.1f}% ({concluidas}/{total_fila})")
    
    if pendentes_reais <= 0 and total_fila > 0:
        st.success("Todas as tarefas da fila foram concluídas!")

def tab_processamento(pasta_origem, pasta_destino, tolerancia_match, tamanho_min_face, confianca_minima):
    st.subheader("Controle de Fila e Processamento")
    col_p1, col_p2 = st.columns(2)
    with col_p1: 
        p_origem = st.text_input("Diretório de Origem:", value=pasta_origem, key="cfg_origem")
    with col_p2: 
        p_destino = st.text_input("Diretório de Destino:", value=pasta_destino, key="cfg_destino")

    st.write("")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    with col_btn1:
        if st.button("📋 Alimentar Fila", use_container_width=True):
            if not os.path.exists(p_origem):
                st.error("Diretório não localizado.")
            else:
                with st.spinner("Indexando..."):
                    from maestro_scanner import mapear_lotes_de_extracao
                    total = mapear_lotes_de_extracao(p_origem)
                st.success(f"{total} registros inseridos.")

    with col_btn2:
        if st.button("▶️ Iniciar Processamento", type="primary", use_container_width=True):
            with st.spinner("Despachando..."):
                tarefas = obter_todas_tarefas_pendentes()
                for t in tarefas:
                    tarefa_id, id_extracao, caminho_relativo = t
                    caminho_fisico = os.path.join(p_origem, caminho_relativo)
                    extrair_faces_tarefa.delay(tarefa_id, id_extracao, caminho_fisico, p_destino, tolerancia_match, tamanho_min_face, confianca_minima)
            st.success(f"{len(tarefas)} tarefas enviadas!")

    with col_btn3:
        if st.button("🛑 Parar Processamento", use_container_width=True):
            with st.spinner("Parando..."):
                from worker import app 
                app.control.purge()
                from database_manager import cancelar_tarefas_pendentes
                cancelar_tarefas_pendentes()
            st.warning("Interrompido!")

    with col_btn4:
        if st.button("🔄 Resetar Banco de Dados", use_container_width=True):
            if resetar_banco_dados():
                st.success("Banco de dados resetado com sucesso.")
            else:
                st.error("Falha ao resetar o banco de dados.")

    st.divider()
    monitorar_progresso_fragment()

def tab_auditoria(pasta_destino):
    st.subheader("Auditoria de Reincidentes")
    termo_busca = st.text_input("Filtrar por ID ou Lote de Reincidente:", value="", key="busca_auditoria").strip().lower()
    
    if os.path.exists(pasta_destino):
        ids_suspeitos = sorted([d for d in os.listdir(pasta_destino) if os.path.isdir(os.path.join(pasta_destino, d)) and not d.startswith("_")])
        
        reincidentes = []
        for ind_id in ids_suspeitos:
            caminho_id = os.path.join(pasta_destino, ind_id)
            fotos = sorted([f for f in os.listdir(caminho_id) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            fotos_unicas = []
            nomes_vistos = set()
            for f in fotos:
                tamanho_arq = os.path.getsize(os.path.join(caminho_id, f))
                id_unico = f"{f.split('_')[0]}_{tamanho_arq}"
                if id_unico not in nomes_vistos:
                    nomes_vistos.add(id_unico)
                    fotos_unicas.append(f)

            if len(fotos_unicas) > 1:
                lotes = set([f.split("_")[1] if "celular_" in f and len(f.split("_")) > 1 else "Geral" for f in fotos_unicas])
                if not termo_busca or termo_busca in ind_id.lower() or any(termo_busca in l.lower() for l in lotes):
                    reincidentes.append({"id": ind_id, "caminho": caminho_id, "fotos": fotos_unicas, "lotes": list(lotes)})

        st.markdown(f"**Total de Reincidentes Filtrados: {len(reincidentes)}**")
        
        for item in reincidentes[:25]:
            with st.expander(f"ID: {item['id']} | {len(item['fotos'])} Aparições distintas", expanded=False):
                st.markdown(f"<span class='badge badge-reincidente'>REINCIDENTE</span> &nbsp; Origens: {', '.join(item['lotes'])}", unsafe_allow_html=True)
                cols = st.columns(min(len(item["fotos"]), 6))
                for idx, foto_nome in enumerate(item["fotos"][:6]):
                    with cols[idx]:
                        st.image(os.path.join(item["caminho"], foto_nome), use_container_width=True, caption=foto_nome)
                if len(item["fotos"]) > 6:
                    st.caption(f"+ {len(item['fotos']) - 6} foto(s) ocultada(s) para otimizar desempenho.")
    else:
        st.warning("Diretório de armazenamento não encontrado.")

def tab_busca(confianca_minima, tolerancia_match, tamanho_min_face, pasta_destino):
    st.subheader("Busca Biométrica Avançada")
    col_ref, col_res = st.columns([1, 2])
    
    with col_ref:
        uploaded_file = st.file_uploader("Enviar Imagem de Referência", type=["jpg", "jpeg", "png"], key="up_ref")
        img_bgr = None
        if uploaded_file is not None:
            nparr = np.frombuffer(uploaded_file.read(), np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Alvo da Busca", use_container_width=True)
            executar = st.button("Executar Varredura", type="primary", use_container_width=True)
        else:
            executar = False

    with col_res:
        if executar and img_bgr is not None:
            app_arcface = carregar_modelo_arcface(confianca_minima)
            if app_arcface:
                faces = app_arcface.get(img_bgr)
                faces_validas = [f for f in faces if min(f.bbox[2]-f.bbox[0], f.bbox[3]-f.bbox[1]) >= tamanho_min_face]
                if faces_validas:
                    for face in faces_validas:
                        resultados = buscar_historico_completo_por_vetor(face.embedding.astype(np.float32), limite_distancia=tolerancia_match)
                        if resultados:
                            st.success(f"{len(resultados)} correspondência(s) encontrada(s)!")
                            for res in resultados:
                                acc = max(0, (1 - res['distancia']) * 100)
                                ind_id = res['individuo_id']
                                
                                pasta_ind = os.path.join(pasta_destino, ind_id)
                                foto_referencia = None
                                det_origem = "Desconhecida"
                                
                                if os.path.exists(pasta_ind):
                                    fotos_banco = sorted([f for f in os.listdir(pasta_ind) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                                    if fotos_banco:
                                        foto_referencia = os.path.join(pasta_ind, fotos_banco[0])
                                        det_origem = fotos_banco[0]

                                with st.container():
                                    st.markdown(f"""<div class="result-card">""", unsafe_allow_html=True)
                                    col_m1, col_m2 = st.columns([1, 4])
                                    with col_m1:
                                        if foto_referencia and os.path.exists(foto_referencia):
                                            st.image(foto_referencia, use_container_width=True)
                                        else:
                                            st.info("Sem foto.")
                                    with col_m2:
                                        st.markdown(f"""
                                        <h4 style="margin:0; color:#60a5fa;">ID Localizado: {ind_id}</h4>
                                        <p style="margin:5px 0 0 0;">Acurácia: <b style="color:#10b981;">{acc:.2f}%</b> | Distância: {res['distancia']:.4f}</p>
                                        <p style="margin:5px 0 0 0; font-size:0.85rem; color:#94a3b8;"><b>Origem Extraída:</b> {det_origem}</p>
                                        <p style="margin:2px 0 0 0; font-size:0.85rem; color:#94a3b8;"><b>Caminho no Banco:</b> {pasta_ind}</p>
                                        """, unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("Nenhuma correspondência encontrada com os parâmetros de tolerância atuais.")
                else:
                    st.warning("Nenhum rosto válido detectado na imagem referência.")

def tab_curadoria(pasta_destino):
    st.subheader("Curadoria e Análise de Descartes")
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("##### 📉 Descarte por Baixa Confiança")
        p_conf = os.path.join(pasta_destino, "_descartados_confianca")
        if os.path.exists(p_conf):
            fotos = os.listdir(p_conf)[:6]
            if fotos:
                grid = st.columns(3)
                for i, f in enumerate(fotos):
                    with grid[i % 3]:
                        st.image(os.path.join(p_conf, f), use_container_width=True)
            else:
                st.info("Nenhum registro encontrado.")
                
    with col_c2:
        st.markdown("##### 📏 Descarte por Tamanho Insuficiente")
        p_tam = os.path.join(pasta_destino, "_descartados_tamanho")
        if os.path.exists(p_tam):
            fotos = os.listdir(p_tam)[:6]
            if fotos:
                grid = st.columns(3)
                for i, f in enumerate(fotos):
                    with grid[i % 3]:
                        st.image(os.path.join(p_tam, f), use_container_width=True)
            else:
                st.info("Nenhum registro encontrado.")

def carregar_painel():
    col_header, col_logo = st.columns([8, 1])
    
    with col_header:
        st.markdown("""
            <div class="pericia-title-container" style="padding-top: 10px;">
                <h1 class="pericia-title">PERÍCIA FACE | Polícia Científica</h1>
                <p class="pericia-subtitle">Sistema unificado de extração biométrica e varredura pericial automatizada.</p>
            </div>
        """, unsafe_allow_html=True)

    with col_logo:
        caminho_logo = "29fc63f478b81ace491a329a223c76a6.jpg"
        if os.path.exists(caminho_logo):
            st.markdown('<div class="logo-container">', unsafe_allow_html=True)
            st.image(caminho_logo, width=120)
            st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 25px; border: 1px solid #1e293b;'/>", unsafe_allow_html=True)

    st.sidebar.markdown("### ⚙️ Calibração Forense")
    confianca_minima = st.sidebar.slider("Confiança Mínima", 0.15, 0.40, 0.20, 0.01)
    tolerancia_match = st.sidebar.slider("Tolerância Match", 0.35, 0.65, 0.45, 0.01)
    tamanho_min_face = st.sidebar.slider("Tamanho Mínimo", 15, 50, 20, 1)
    
    pasta_origem = os.getenv("PASTA_ORIGEM", r"C:\Users\saulo.borges\Downloads\facial_teste\entradas_celular")
    pasta_destino = os.getenv("PASTA_DESTINO", r"C:\Users\saulo.borges\Downloads\facial_teste\armazenamento_crops")

    a1, a2, a3, a4 = st.tabs(["Processamento de Fila", "Auditoria de Reincidência", "Busca Biométrica", "Curadoria de Descartes"])
    
    with a1:
        tab_processamento(pasta_origem, pasta_destino, tolerancia_match, tamanho_min_face, confianca_minima)
    with a2:
        tab_auditoria(pasta_destino)
    with a3:
        tab_busca(confianca_minima, tolerancia_match, tamanho_min_face, pasta_destino)
    with a4:
        tab_curadoria(pasta_destino)

if __name__ == "__main__":
    carregar_painel()