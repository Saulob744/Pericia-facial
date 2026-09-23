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
    pegar_tarefa_pendente,
    obter_reincidentes_banco
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
    .result-card { background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #3b82f6; margin-bottom: 20px; }
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
    col_p1, col_p2, col_vazia = st.columns([2, 2, 3])
    with col_p1: 
        input_origem = st.text_input("Diretório de Origem:", placeholder="[Caminho Padrão Oculto por Segurança] - Digite para alterar", value="")
        p_origem_real = input_origem.strip() if input_origem.strip() else pasta_origem
        
    with col_p2: 
        input_destino = st.text_input("Diretório de Destino:", placeholder="[Caminho Padrão Oculto por Segurança] - Digite para alterar", value="")
        p_destino_real = input_destino.strip() if input_destino.strip() else pasta_destino

    st.write("")
    col_btn1, col_btn2, col_btn3, col_btn4, col_spacer = st.columns([1.5, 1.5, 1.5, 1.5, 4])

    with col_btn1:
        if st.button("Alimentar Fila", use_container_width=True):
            if not os.path.exists(p_origem_real):
                st.error("Diretório não localizado.")
            else:
                with st.spinner("Indexando..."):
                    from maestro_scanner import mapear_lotes_de_extracao
                    total = mapear_lotes_de_extracao(p_origem_real)
                st.success(f"{total} registros inseridos.")

    with col_btn2:
        if st.button("▶️ Iniciar", type="primary", use_container_width=True):
            with st.spinner("Despachando..."):
                tarefas = obter_todas_tarefas_pendentes()
                for t in tarefas:
                    tarefa_id, id_extracao, caminho_relativo = t
                    caminho_fisico = os.path.join(p_origem_real, caminho_relativo)
                    extrair_faces_tarefa.delay(tarefa_id, id_extracao, caminho_fisico, p_destino_real, tolerancia_match, tamanho_min_face, confianca_minima)
            st.success(f"{len(tarefas)} tarefas enviadas!")

    with col_btn3:
        if st.button("🛑 Parar", use_container_width=True):
            with st.spinner("Parando..."):
                from worker import app 
                app.control.purge()
                from database_manager import cancelar_tarefas_pendentes
                cancelar_tarefas_pendentes()
            st.warning("Interrompido!")

    with col_btn4:
        if st.button("🔄 Resetar Banco", use_container_width=True):
            if resetar_banco_dados():
                st.success("Banco resetado com sucesso.")
            else:
                st.error("Falha ao resetar banco.")

    st.divider()
    monitorar_progresso_fragment()

def tab_auditoria():
    st.subheader("Auditoria de Reincidentes")
    
    if 'limite_auditoria' not in st.session_state:
        st.session_state.limite_auditoria = 25
        
    def carregar_mais_reincidentes():
        st.session_state.limite_auditoria += 25

    termo_busca = st.text_input("Filtrar por ID ou Lote de Reincidente:", value="", key="busca_auditoria").strip().lower()
    
    reincidentes_banco = obter_reincidentes_banco()
    
    reincidentes = []
    for item in reincidentes_banco:
        if not termo_busca or termo_busca in item["id"].lower() or any(termo_busca in lote.lower() for lote in item["lotes"]):
            reincidentes.append(item)

    st.markdown(f"**Total de Reincidentes Filtrados: {len(reincidentes)}**")
    
    for item in reincidentes[:st.session_state.limite_auditoria]:
        with st.expander(f"ID: {item['id']} | {len(item['fotos'])} Aparições distintas", expanded=False):
            st.markdown(f"<span class='badge badge-reincidente'>REINCIDENTE</span> &nbsp; Origens: {', '.join(item['lotes'])}", unsafe_allow_html=True)
            st.write("")
            
            qtd_miniaturas = 12
            cols = st.columns(qtd_miniaturas)
            
            for idx, foto_path in enumerate(item["fotos"][:qtd_miniaturas]):
                with cols[idx]:
                    if os.path.exists(foto_path):
                        st.image(foto_path, use_container_width=True)
                    else:
                        st.error("X")
                        
            if len(item["fotos"]) > qtd_miniaturas:
                st.caption(f"+ {len(item['fotos']) - qtd_miniaturas} foto(s) ocultada(s) para otimizar desempenho.")
    
    if st.session_state.limite_auditoria < len(reincidentes):
        st.write("")
        col_vazia1, col_botao, col_vazia2 = st.columns([1.5, 1, 1.5])
        with col_botao:
            st.button(
                "⬇️ CARREGAR MAIS RESULTADOS", 
                on_click=carregar_mais_reincidentes,
                type="primary",
                use_container_width=True
            )

def tab_busca(confianca_minima, tolerancia_match, tamanho_min_face, pasta_destino):
    st.subheader("Busca Biométrica Avançada")
    
    def formatar_caminho_seguro(caminho):
        if not caminho: return "Desconhecido"
        c = caminho.replace('\\', '/')
        parts = c.split('/')
        parts_lower = [p.lower() for p in parts]
        if 'downloads' in parts_lower:
            idx = parts_lower.index('downloads')
            return ".../" + "/".join(parts[idx:])
        return ".../" + "/".join(parts[-3:]) if len(parts) > 3 else c

    uploaded_file = st.file_uploader("Enviar Imagem de Referência", type=["jpg", "jpeg", "png"], key="up_ref")
    
    if uploaded_file is not None:
        nparr = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        col_esquerda, col_direita = st.columns([1, 2.5])
        
        with col_esquerda:
            st.markdown("##### 🎯 Alvo da Busca")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            executar = st.button("🔍 Executar Varredura", type="primary", use_container_width=True)

        with col_direita:
            if executar:
                st.markdown("##### 🔎 Resultados Encontrados")
                app_arcface = carregar_modelo_arcface(confianca_minima)
                if app_arcface:
                    faces = app_arcface.get(img_bgr)
                    faces_validas = [f for f in faces if min(f.bbox[2]-f.bbox[0], f.bbox[3]-f.bbox[1]) >= tamanho_min_face]
                    
                    if faces_validas:
                        for face in faces_validas:
                            resultados = buscar_historico_completo_por_vetor(face.embedding.astype(np.float32), limite_distancia=tolerancia_match)
                            
                            if resultados:
                                st.success(f"{len(resultados)} correspondência(s) encontrada(s) no banco de dados!")
                                
                                for res in resultados:
                                    acc = max(0, (1 - res['distancia']) * 100)
                                    ind_id = res['individuo_id']
                                    ocorrencias = res.get('ocorrencias', [])
                                    
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <h4 style="margin:0 0 5px 0; color:#60a5fa;">ID Localizado: {ind_id}</h4>
                                        <p style="margin:0 0 15px 0;">Acurácia Biométrica: <b style="color:#10b981; font-size:1.1rem;">{acc:.2f}%</b> | Distância: {res['distancia']:.4f}</p>
                                    """, unsafe_allow_html=True)
                                    
                                    if ocorrencias:
                                        lotes_extras = list(set([oc['lote'] for oc in ocorrencias]))
                                        if len(lotes_extras) > 1:
                                            st.markdown(f"<p style='margin:0 0 10px 0; font-size:0.9rem; color:#f59e0b;'>⚠️ Detecções em {len(lotes_extras)} lotes distintos de extração.</p>", unsafe_allow_html=True)
                                        
                                        thumb_cols = st.columns(6) 
                                        for idx, oc in enumerate(ocorrencias):
                                            path_oc = oc.get('caminho_crop')
                                            path_orig = oc.get('caminho_arquivo', '')
                                            data_oc = oc.get('data_ocorrencia', '')
                                            
                                            str_data = data_oc.strftime('%d/%m/%Y %H:%M') if hasattr(data_oc, 'strftime') else str(data_oc).split('.')[0]
                                            caminho_limpo = formatar_caminho_seguro(path_orig)
                                            
                                            if path_oc and os.path.exists(path_oc):
                                                with thumb_cols[idx % 6]:
                                                    st.image(path_oc, use_container_width=True)
                                                    st.markdown(f"""
                                                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px; margin-bottom: 15px; word-wrap: break-word; line-height: 1.3;">
                                                        <b style="color: #cbd5e1;">Lote:</b> {oc.get('lote')}<br>
                                                        <b style="color: #cbd5e1;">Data:</b> {str_data}<br>
                                                        <b style="color: #cbd5e1;">Origem:</b> {caminho_limpo}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.info("Nenhuma correspondência encontrada com os parâmetros de tolerância atuais.")
                    else:
                        st.warning("Nenhum rosto válido (ou com tamanho suficiente) detectado na imagem referência.")
            else:
                st.info("Clique em 'Executar Varredura' para processar a imagem e buscar no banco de dados.")

def tab_curadoria(pasta_destino):
    st.subheader("Curadoria e Análise de Descartes")
    
    st.markdown("##### Galeria de Imagens Descartadas (Sem Rosto ou Tamanho Insuficiente)")
    p_descartes = os.path.join(pasta_destino, "_descartados_sem_rosto")
    
    if 'limite_descartes' not in st.session_state:
        st.session_state.limite_descartes = 24

    def carregar_mais_descartes():
        st.session_state.limite_descartes += 24

    if os.path.exists(p_descartes):
        todas_fotos = [f for f in os.listdir(p_descartes) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_descartes = len(todas_fotos)
        
        if total_descartes > 0:
            st.markdown(f"**Total de imagens descartadas:** {total_descartes}")
            st.write("")
            
            fotos_exibidas = todas_fotos[:st.session_state.limite_descartes]
            cols = st.columns(8)
            
            for i, f in enumerate(fotos_exibidas):
                with cols[i % 8]:
                    caminho_foto = os.path.join(p_descartes, f)
                    st.image(caminho_foto, use_container_width=True)
            
            if st.session_state.limite_descartes < total_descartes:
                st.write("")
                col_vazia1, col_botao, col_vazia2 = st.columns([1.5, 1, 1.5])
                with col_botao:
                    st.button(
                        "⬇️ CARREGAR MAIS DESCARTES", 
                        on_click=carregar_mais_descartes,
                        key="btn_mais_descartes",
                        type="primary",
                        use_container_width=True
                    )
        else:
            st.info("A pasta de descartes está vazia no momento.")
    else:
        st.info("A pasta de descartes ainda não foi criada pelo sistema.")

def carregar_painel():
    col_header, col_logo = st.columns([7.5, 2.5])
    
    with col_header:
        st.markdown("""
            <div class="pericia-title-container" style="padding-top: 10px;">
                <h1 class="pericia-title">PERÍCIA FACE | Polícia Científica</h1>
                <p class="pericia-subtitle">Sistema unificado de extração biométrica e varredura pericial automatizada.</p>
            </div>
        """, unsafe_allow_html=True)

    with col_logo:
        caminho_logo1 = "29fc63f478b81ace491a329a223c76a6.jpg"
        caminho_logo2 = "Gemini_Generated_Image_5u7pwj5u7pwj5u7p.jpeg"
        
        import base64
        def obter_base64(caminho):
            with open(caminho, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        html_logos = '<div style="display: flex; justify-content: flex-end; align-items: center; gap: 15px; height: 100%;">'
        if os.path.exists(caminho_logo1):
            img1_b64 = obter_base64(caminho_logo1)
            html_logos += f'<img src="data:image/jpeg;base64,{img1_b64}" style="height: 150px; width: auto; object-fit: contain; border-radius: 8px;">'
        if os.path.exists(caminho_logo2):
            img2_b64 = obter_base64(caminho_logo2)
            html_logos += f'<img src="data:image/jpeg;base64,{img2_b64}" style="height: 150px; width: auto; object-fit: contain; border-radius: 8px;">'
            
        html_logos += '</div>'
        
        st.markdown(html_logos, unsafe_allow_html=True)
            
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 25px; border: 1px solid #1e293b;'/>", unsafe_allow_html=True)

    st.sidebar.markdown("### ⚙️ Padrões Forenses")
    
    confianca_minima = 0.25
    tolerancia_match = 0.60
    tamanho_min_face = 40
    
    st.sidebar.info(f"""
    **Parâmetros travados pelo sistema:**
    - Confiança de Detecção: **{confianca_minima}**
    - Tolerância de Biometria: **{tolerancia_match}**
    - Rosto Mínimo: **{tamanho_min_face}px**
    """)
    
    pasta_origem = os.getenv("PASTA_ORIGEM", r"C:\Users\saulo.borges\Downloads\facial_teste\entradas_celular")
    pasta_destino = os.getenv("PASTA_DESTINO", r"C:\Users\saulo.borges\Downloads\facial_teste\armazenamento_crops")

    a1, a2, a3, a4 = st.tabs(["Processamento de Fila", "Auditoria de Reincidência", "Busca Biométrica", "Curadoria de Descartes"])
    
    with a1:
        tab_processamento(pasta_origem, pasta_destino, tolerancia_match, tamanho_min_face, confianca_minima)
    with a2:
        tab_auditoria()
    with a3:
        tab_busca(confianca_minima, tolerancia_match, tamanho_min_face, pasta_destino)
    with a4:
        tab_curadoria(pasta_destino)

if __name__ == "__main__":
    carregar_painel()