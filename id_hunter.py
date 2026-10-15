import base64
import os
import shutil
import warnings
import cv2
import numpy as np
import streamlit as st
import json
from datetime import datetime
import stat

METRICAS_FILE = "/app/data/metricas_teste.json"

def salvar_inicio_teste():
    dados = {
        "inicio_iso": datetime.now().isoformat(),
        "status": "EM_ANDAMENTO"
    }
    try:
        os.makedirs(os.path.dirname(METRICAS_FILE), exist_ok=True)
        with open(METRICAS_FILE, "w") as f:
            json.dump(dados, f)
    except Exception:
        pass

def obter_inicio_teste():
    if os.path.exists(METRICAS_FILE):
        try:
            with open(METRICAS_FILE, "r") as f:
                dados = json.load(f)
                return datetime.fromisoformat(dados["inicio_iso"])
        except Exception:
            return None
    return None

def limpar_metricas_teste():
    if os.path.exists(METRICAS_FILE):
        try:
            os.remove(METRICAS_FILE)
        except Exception:
            pass

from database_manager import (
    buscar_historico_completo_por_vetor,
    obter_estatisticas_sistema,
    obter_reincidentes_banco,
    resetar_banco_dados,
    pegar_tarefas_em_lote,
    cancelar_tarefas_pendentes,
)
from worker import processar_lote_faces_tarefa, app as celery_app

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(layout="wide", page_title="PERÍCIA FACE | Polícia Científica", page_icon="🛡️")

# Estilização CSS do Painel
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
    div[data-baseweb="tab-list"] {
        justify-content: flex-start !important;
        gap: 16px !important;
        margin-bottom: 25px !important;
    }
    button[data-baseweb="tab"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        border-radius: 8px 8px 0px 0px !important;
        background-color: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
    }
    button[aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }
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


def obter_base64(caminho):
    if not os.path.exists(caminho):
        return ""
    with open(caminho, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def formatar_caminho_seguro(caminho):
    if not caminho:
        return "Desconhecido"
    c = caminho.replace('\\', '/')
    parts = c.split('/')
    return ".../" + "/".join(parts[-3:]) if len(parts) > 3 else c


def remove_readonly(func, path, excinfo):
    """Garante a remoção de arquivos somente-leitura ou protegidos pelo Docker."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def limpar_arquivos_gerados(pasta_destino):
    if not os.path.exists(pasta_destino):
        return True, "Diretório de destino está vazio ou não existe."

    erros = 0
    removidos = 0

    for item in os.listdir(pasta_destino):
        item_path = os.path.join(pasta_destino, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.chmod(item_path, stat.S_IWRITE)
                os.unlink(item_path)
                removidos += 1
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path, onerror=remove_readonly)
                removidos += 1
        except Exception:
            erros += 1

    if erros > 0:
        return False, f"Removidos {removidos} itens, mas {erros} falharam por permissão do Docker."
    
    return True, f"Sucesso! {removidos} pasta(s)/arquivo(s) de crops foram limpos do disco."

@st.fragment(run_every="3s")
def monitorar_progresso_fragment():
    stats = obter_estatisticas_sistema()
    fila_stats = stats.get("fila", {})
    bio_stats = stats.get("biometria", {})
    pendentes_reais = fila_stats.get("PENDENTE", 0) + fila_stats.get("PROCESSANDO", 0)
    concluidas = fila_stats.get("CONCLUIDO", 0) + fila_stats.get("SEM_ROSTO", 0) + fila_stats.get("FALHA", 0)
    total_valido = pendentes_reais + concluidas
    
    progresso_pct = (concluidas / total_valido) if total_valido > 0 else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card" style="border-top: 4px solid #3b82f6;"><p class="metric-val" style="color:#60a5fa;">{pendentes_reais}</p><p class="metric-lbl">Pendentes</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="border-top: 4px solid #10b981;"><p class="metric-val" style="color:#34d399;">{fila_stats.get("CONCLUIDO", 0)}</p><p class="metric-lbl">Faces Extraídas</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="border-top: 4px solid #f59e0b;"><p class="metric-val" style="color:#fbbf24;">{fila_stats.get("SEM_ROSTO", 0)}</p><p class="metric-lbl">Sem Face</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card" style="border-top: 4px solid #ef4444;"><p class="metric-val" style="color:#f87171;">{bio_stats.get("total_individuos", 0)}</p><p class="metric-lbl">Cadastros Únicos</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card" style="border-top: 4px solid #8b5cf6;"><p class="metric-val" style="color:#a78bfa;">{bio_stats.get("total_ocorrencias", 0)}</p><p class="metric-lbl">Total Ocorrências</p></div>', unsafe_allow_html=True)

    st.write("")
    st.progress(progresso_pct, text=f"Progresso de Processamento: {progresso_pct*100:.1f}% ({concluidas}/{total_fila})")

    inicio_dt = obter_inicio_teste()
    if inicio_dt and concluidas > 0:
        decorrido_seg = max(1, (datetime.now() - inicio_dt).total_seconds())
        v_fps = concluidas / decorrido_seg
        v_fpm = v_fps * 60
        
        m_dec, s_dec = divmod(int(decorrido_seg), 60)
        h_dec, m_dec = divmod(m_dec, 60)
        str_decorrido = f"{h_dec:02d}h {m_dec:02d}m {s_dec:02d}s"
        
        if pendentes_reais > 0 and v_fps > 0:
            eta_seg = int(pendentes_reais / v_fps)
            m_eta, s_eta = divmod(eta_seg, 60)
            h_eta, m_eta = divmod(m_eta, 60)
            str_eta = f"{h_eta:02d}h {m_eta:02d}m {s_eta:02d}s"
        else:
            str_eta = "Concluído" if pendentes_reais == 0 else "Calculando..."

        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 15px 20px; border-radius: 10px; border: 1px solid #334155; margin-top: 15px;">
            <h5 style="margin: 0 0 10px 0; color: #38bdf8;">📊 Relatório de Performance em Tempo Real</h5>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; font-size: 0.95rem;">
                <div><b>🕒 Horário de Início:</b> <span style="color: #cbd5e1;">{inicio_dt.strftime('%H:%M:%S (%d/%m)')}</span></div>
                <div><b>⏱️ Tempo Decorrido:</b> <span style="color: #fbbf24;">{str_decorrido}</span></div>
                <div><b>⚡ Velocidade Média:</b> <span style="color: #34d399;">{v_fps:.2f} fotos/s</span> ({v_fpm:.1f} fotos/min)</div>
                <div><b>⏳ Previsão de Término (ETA):</b> <span style="color: #f87171;">{str_eta}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if pendentes_reais <= 0 and total_fila > 0:
        st.success("Todas as tarefas da fila foram concluídas!")


def tab_processamento(pasta_origem, pasta_destino, tolerancia_match, tamanho_min_face, confianca_minima):
    st.subheader("Controle de Fila e Processamento")
    col_p1, col_p2, _ = st.columns([3, 3, 1])
    
    with col_p1: 
        input_origem = st.text_input("Diretório de Origem (Entradas):", value=pasta_origem)
        p_origem_real = input_origem.strip() if input_origem.strip() else pasta_origem
        
    with col_p2: 
        input_destino = st.text_input("Diretório de Destino (Crops/Resultados):", value=pasta_destino)
        p_destino_real = input_destino.strip() if input_destino.strip() else pasta_destino

    st.write("")
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([1.5, 1.5, 1.5, 1.8, 1.8])

    with col_btn1:
        if st.button("📥 Alimentar Fila", use_container_width=True):
            if not os.path.exists(p_origem_real):
                st.error(f"Diretório não localizado: {p_origem_real}")
            else:
                with st.spinner("Indexando arquivos..."):
                    from maestro_scanner import mapear_lotes_de_extracao
                    total = mapear_lotes_de_extracao(p_origem_real)
                st.success(f"{total} registros inseridos na fila.")

    with col_btn2:
            if st.button("▶️ Iniciar Lotes", type="primary", use_container_width=True):
                salvar_inicio_teste()
                if not os.path.exists(p_origem_real):
                    st.error(f"Diretório de origem inválido: {p_origem_real}")
                else:
                    with st.spinner("Despachando lotes vetorizados para os workers..."):
                        total_enviadas = 0
                        tamanho_lote = 32

                        while True:
                            tarefas = pegar_tarefas_em_lote(tamanho_lote)
                            if not tarefas:
                                break
                            
                            lote_envio = []
                            for t in tarefas:
                                tarefa_id, id_extracao, caminho_relativo = t
                                caminho_fisico = os.path.join(p_origem_real, caminho_relativo) if not os.path.isabs(caminho_relativo) else caminho_relativo
                                
                                lote_envio.append({
                                    "tarefa_id": tarefa_id,
                                    "id_extracao": id_extracao,
                                    "caminho_fisico": caminho_fisico,
                                    "pasta_destino": p_destino_real,
                                    "tolerancia_match": tolerancia_match,
                                    "tamanho_min_face": tamanho_min_face,
                                    "confianca_minima": confianca_minima
                                })
                            
                            processar_lote_faces_tarefa.delay(lote_envio)
                            total_enviadas += len(lote_envio)
                            
                    st.success(f"Sucesso! {total_enviadas} tarefas foram despachadas em lotes de {tamanho_lote}.")

    with col_btn3:
        if st.button("🛑 Parar", use_container_width=True):
            with st.spinner("Expurgando fila do Celery e cancelando tarefas..."):
                celery_app.control.purge()
                cancelar_tarefas_pendentes()
                limpar_metricas_teste()
            st.warning("Processamento interrompido!")
            st.rerun()

    with col_btn4:
        if st.button("🔄 Resetar Banco", use_container_width=True):
            limpar_metricas_teste() 
            if resetar_banco_dados():
                st.success("Banco de dados resetado com sucesso.")

    with col_btn5:
        if st.button("🗑️ Limpar Crops", use_container_width=True):
            sucesso, mensagem = limpar_arquivos_gerados(p_destino_real)
            monitorar_progresso_fragment()
            if sucesso:
                st.success(mensagem)
                st.rerun()
            else:
                st.warning(mensagem)
    


@st.fragment(run_every="5s")
def renderizar_auditoria_fragment(termo_busca):
    reincidentes_banco = obter_reincidentes_banco()
    
    reincidentes = [
        item for item in reincidentes_banco
        if not termo_busca or termo_busca in item["id"].lower() or any(termo_busca in lote.lower() for lote in item["lotes"])
    ]

    st.markdown(f"**Total de Reincidentes Filtrados: {len(reincidentes)}**")
    
    for item in reincidentes[:st.session_state.limite_auditoria]:
        with st.expander(f"ID: {item['id']} | {len(item['fotos'])} Aparições distintas", expanded=False):
            st.markdown(f"<span class='badge badge-reincidente'>REINCIDENTE</span> &nbsp; Origens: {', '.join(item['lotes'])}", unsafe_allow_html=True)
            st.write("")
            
            qtd_miniaturas = 12
            cols = st.columns(qtd_miniaturas)
            
            for idx, foto_path in enumerate(item["fotos"][:qtd_miniaturas]):
                with cols[idx]:
                    if os.path.exists(foto_path) and os.path.getsize(foto_path) > 0:
                        try:
                            st.image(foto_path, use_container_width=True)
                        except Exception:
                            st.error("X")
                    else:
                        st.error("X")
                        
            if len(item["fotos"]) > qtd_miniaturas:
                st.caption(f"+ {len(item['fotos']) - qtd_miniaturas} foto(s) ocultada(s) para otimizar desempenho.")
    
    if st.session_state.limite_auditoria < len(reincidentes):
        st.write("")
        _, col_botao, _ = st.columns([1.5, 1, 1.5])
        with col_botao:
            st.button(
                "⬇️ CARREGAR MAIS RESULTADOS", 
                on_click=lambda: st.session_state.update(limite_auditoria=st.session_state.limite_auditoria + 25),
                type="primary",
                use_container_width=True,
                key="btn_carregar_mais_auditoria"
            )


def tab_auditoria():
    st.subheader("Auditoria de Reincidentes")
    
    if 'limite_auditoria' not in st.session_state:
        st.session_state.limite_auditoria = 25

    col_f, _ = st.columns([2, 2])
    with col_f:
        termo_busca = st.text_input("Filtrar por ID ou Lote de Reincidente:", value="", key="busca_auditoria").strip().lower()

    renderizar_auditoria_fragment(termo_busca)


def tab_busca(confianca_minima, tolerancia_match, tamanho_min_face, pasta_destino):
    st.subheader("Busca Biométrica Avançada")

    col_up, _ = st.columns([2, 2])
    with col_up:
        uploaded_file = st.file_uploader("Enviar Imagem de Referência", type=["jpg", "jpeg", "png", "heic", "heif"], key="up_ref")
    
    if uploaded_file is not None:
        nparr = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        col_esquerda, col_direita = st.columns([1, 2.5])
        
        with col_esquerda:
            st.markdown("##### 🎯 Alvo da Busca")
            if img_bgr is not None:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            executar = st.button("🔍 Executar Varredura", type="primary", use_container_width=True)

        with col_direita:
            if executar and img_bgr is not None:
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
                                            
                                            if path_oc and os.path.exists(path_oc) and os.path.getsize(path_oc) > 0:
                                                with thumb_cols[idx % 6]:
                                                    try:
                                                        st.image(path_oc, use_container_width=True)
                                                    except Exception:
                                                        pass
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
            elif not executar:
                st.info("Clique em 'Executar Varredura' para processar a imagem e buscar no banco de dados.")


@st.fragment(run_every="5s")
def renderizar_galeria_descartes(pasta, chave_sessao, key_prefix):
    extensoes = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.heic', '.heif')
    if chave_sessao not in st.session_state:
        st.session_state[chave_sessao] = 24

    if os.path.exists(pasta):
        fotos = [f for f in os.listdir(pasta) if f.lower().endswith(extensoes)]
        total = len(fotos)
        
        if total > 0:
            st.markdown(f"**Total descartado: {total}**")
            st.write("")
            
            cols = st.columns(8)
            for i, f in enumerate(fotos[:st.session_state[chave_sessao]]):
                caminho_img = os.path.join(pasta, f)
                if os.path.getsize(caminho_img) > 0:
                    with cols[i % 8]:
                        try:
                            st.image(caminho_img, use_container_width=True)
                        except Exception:
                            pass
            
            if st.session_state[chave_sessao] < total:
                st.write("")
                _, col_b, _ = st.columns([1.5, 1, 1.5])
                with col_b:
                    st.button(
                        "⬇️ CARREGAR MAIS", 
                        on_click=lambda: st.session_state.update({chave_sessao: st.session_state[chave_sessao] + 24}),
                        key=f"btn_{key_prefix}",
                        type="primary",
                        use_container_width=True
                    )
        else:
            st.info("A pasta de descartes está vazia.")
    else:
        st.info("A pasta de descartes ainda não foi criada.")


def tab_curadoria(pasta_destino):
    st.subheader("Curadoria e Análise de Descartes")
    
    cur1, cur2 = st.tabs(["📏 Tamanho Insuficiente", "📉 Baixa Confiança / Sem Rosto"])
    
    with cur1:
        p_tam = os.path.join(pasta_destino, "_descartados_tamanho")
        renderizar_galeria_descartes(p_tam, "limite_descartes_tam", "descartes_tam")

    with cur2:
        p_conf = os.path.join(pasta_destino, "_descartados_sem_rosto")
        p_conf_alt = os.path.join(pasta_destino, "_descartados_confianca")
        p_alvo = p_conf if os.path.exists(p_conf) else p_conf_alt
        renderizar_galeria_descartes(p_alvo, "limite_descartes_conf", "descartes_conf")


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
        
        html_logos = '<div style="display: flex; justify-content: flex-end; align-items: center; gap: 15px; height: 100%;">'
        
        b64_1 = obter_base64(caminho_logo1)
        if b64_1:
            html_logos += f'<img src="data:image/jpeg;base64,{b64_1}" style="height: 150px; width: auto; object-fit: contain; border-radius: 8px;">'
            
        b64_2 = obter_base64(caminho_logo2)
        if b64_2:
            html_logos += f'<img src="data:image/jpeg;base64,{b64_2}" style="height: 150px; width: auto; object-fit: contain; border-radius: 8px;">'
            
        html_logos += '</div>'
        st.markdown(html_logos, unsafe_allow_html=True)
            
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 25px; border: 1px solid #1e293b;'/>", unsafe_allow_html=True)

    st.sidebar.markdown("### ⚙️ Padrões Forenses")
    
    confianca_minima = 0.65
    tolerancia_match = 0.45
    tamanho_min_face = 60
    
    st.sidebar.info(f"""
    **Parâmetros travados pelo sistema:**
    - Confiança de Detecção: **{confianca_minima}**
    - Tolerância de Biometria: **{tolerancia_match}**
    - Rosto Mínimo: **{tamanho_min_face}px**
    """)
    
    # Caminhos padrão para ambiente Docker/Linux (substituindo padrões hardcoded de Windows)
    pasta_origem = os.getenv("PASTA_ORIGEM", "/app/data/entradas_celular")
    pasta_destino = os.getenv("PASTA_DESTINO", "/app/data/armazenamento_crops")

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