import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import warnings
import time
import hashlib
import gc

from database_manager import (
    enfileirar_tarefa,
    pegar_tarefa_pendente,
    atualizar_status_tarefa,
    registrar_face,
    obter_estatisticas_sistema,
    buscar_historico_completo_por_vetor,
    resetar_banco_dados
)

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(
    layout="wide",
    page_title="Facial-Científica | Central Unificada",
    page_icon="🔬"
)

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    p, span, label, div { color: #cbd5e1; }
    .stMarkdown p { color: #cbd5e1 !important; font-size: 0.95rem; }
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
    .stImage img {
        border-radius: 8px;
        border: 1px solid #334155;
        object-fit: cover;
        max-height: 150px;
        transition: all 0.2s ease-in-out;
    }
    .stImage img:hover {
        transform: scale(1.02);
        border-color: #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

if "worker_rodando" not in st.session_state:
    st.session_state["worker_rodando"] = False
if "parar_solicitado" not in st.session_state:
    st.session_state["parar_solicitado"] = False
if "ultimos_crops" not in st.session_state:
    st.session_state["ultimos_crops"] = []

@st.cache_resource
def carregar_modelo_arcface(confianca_atual):
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=confianca_atual)          
        return app
    except Exception as e:
        st.error(f"Erro ao inicializar o motor ArcFace: {e}")
        return None

def calcular_sha256(caminho_arquivo: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(caminho_arquivo, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

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

    aba1, aba2, aba3 = st.tabs([
        "🔬 Processamento e Fila",
        "📂 Auditoria e Reincidência",
        "🔎 Busca Biométrica"
    ])

    with aba1:
        st.subheader("Processamento de Evidências")
        st.caption("Controle de ingestão de arquivos e execução de extração facial.")

        caminho_padrao_origem = r"C:\Users\saulo.borges\Downloads\facial_teste\entradas_celular"
        caminho_padrao_destino = r"C:\Users\saulo.borges\Downloads\facial_teste\armazenamento_crops"
        pasta_rejeitados = r"C:\Users\saulo.borges\Downloads\facial_teste\rejeitados"
        os.makedirs(pasta_rejeitados, exist_ok=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pasta_origem = st.text_input("Diretório de Origem (Lotes):", value=caminho_padrao_origem, key="cfg_origem")
        with col_p2:
            pasta_destino = st.text_input("Diretório de Destino (Crops):", value=caminho_padrao_destino, key="cfg_destino")

        st.write("")

        col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1.2, 1.2])

        with col_btn1:
            if st.button("📋 Alimentar Fila de Processamento", use_container_width=True):
                if not os.path.exists(pasta_origem):
                    st.error(f"Diretório '{pasta_origem}' não localizado.")
                else:
                    with st.spinner("Indexando arquivos via os.scandir() e inserindo em lote no PostgreSQL..."):
                        from maestro_scanner import mapear_lotes_de_extracao
                        total_enfileiradas = mapear_lotes_de_extracao(pasta_origem)
                    st.success(f"Operação concluída. {total_enfileiradas} registros inseridos em lote na fila.")

        with col_btn2:
            btn_iniciar = st.button("▶️ Iniciar Motor de Extração", type="primary", use_container_width=True, disabled=st.session_state["worker_rodando"])
            if btn_iniciar:
                st.session_state["worker_rodando"] = True
                st.session_state["parar_solicitado"] = False

        with col_btn3:
            btn_parar = st.button("⏹️ Interromper Processamento", use_container_width=True, disabled=not st.session_state["worker_rodando"])
            if btn_parar:
                st.session_state["parar_solicitado"] = True
                st.warning("Interrupção agendada. O sistema paralisará após a tarefa atual.")

        with st.expander("⚠️ Manutenção de Banco de Dados (Reset)"):
            st.warning("Esta ação é irreversível e excluirá todo o histórico biométrico, filas e cadastros atuais.")
            confirmar_reset = st.checkbox("Estou ciente e desejo prosseguir com a exclusão dos dados.", key="chk_reset")
            if st.button("Excluir Banco de Dados", type="primary", key="btn_reset_db"):
                if not confirmar_reset:
                    st.error("Confirme a operação marcando a caixa de seleção acima.")
                else:
                    try:
                        resetar_banco_dados()
                        st.session_state["ultimos_crops"] = []
                        st.success("Banco de dados formatado com sucesso.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha operacional: {e}")

        st.divider()

        stats = obter_estatisticas_sistema()
        fila_stats = stats["fila"]
        bio_stats = stats["biometria"]

        total_fila = sum(fila_stats.values())
        concluidas = fila_stats.get("CONCLUIDO", 0) + fila_stats.get("SEM_ROSTO", 0) + fila_stats.get("FALHA", 0)
        progresso_pct = (concluidas / total_fila) if total_fila > 0 else 0.0

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{fila_stats.get("PENDENTE", 0)}</p><p class="metric-lbl">Pendentes</p></div>', unsafe_allow_html=True)
        with col_s2:
            st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#10b981;">{fila_stats.get("CONCLUIDO", 0)}</p><p class="metric-lbl">Faces Extraídas</p></div>', unsafe_allow_html=True)
        with col_s3:
            st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#f59e0b;">{fila_stats.get("SEM_ROSTO", 0)}</p><p class="metric-lbl">Sem Face</p></div>', unsafe_allow_html=True)
        with col_s4:
            st.markdown(f'<div class="metric-card-alert"><p class="metric-val" style="color:#ef4444;">{bio_stats.get("total_individuos", 0)}</p><p class="metric-lbl">Cadastros Únicos</p></div>', unsafe_allow_html=True)
        with col_s5:
            st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{bio_stats.get("total_ocorrencias", 0)}</p><p class="metric-lbl">Total Ocorrências</p></div>', unsafe_allow_html=True)

        st.write("")

        if st.session_state["worker_rodando"]:
            st.info("Motor de extração ativo. Monitorando tarefas...")
            
            prog_bar = st.progress(progresso_pct, text=f"Progresso de Processamento: {progresso_pct*100:.1f}% ({concluidas}/{total_fila})")
            status_live = st.status("Processando...", expanded=True)
            gallery_placeholder = st.empty()

            app_arcface = carregar_modelo_arcface(confianca_minima)

            while st.session_state["worker_rodando"]:
                if st.session_state["parar_solicitado"]:
                    status_live.update(label="Processamento interrompido pelo operador.", state="error")
                    st.session_state["worker_rodando"] = False
                    st.session_state["parar_solicitado"] = False
                    st.rerun()
                    break

                tarefa = pegar_tarefa_pendente()
                if not tarefa:
                    status_live.update(label="Fila de processamento esgotada.", state="complete")
                    st.session_state["worker_rodando"] = False
                    st.success("Todas as tarefas foram concluídas.")
                    break

                tarefa_id, id_extracao, caminho_relativo = tarefa
                caminho_fisico = os.path.join(pasta_origem, caminho_relativo)

                status_live.write(f"**ID:** {tarefa_id} | **Lote:** {id_extracao} | **Arquivo:** {caminho_relativo}")

                hash_base = calcular_sha256(caminho_fisico)
                if not hash_base or not os.path.exists(caminho_fisico):
                    atualizar_status_tarefa(tarefa_id, 'FALHA')
                    continue

                img_bgr = cv2.imread(caminho_fisico)
                if img_bgr is None:
                    atualizar_status_tarefa(tarefa_id, 'FALHA')
                    continue

                faces = app_arcface.get(img_bgr)

                if len(faces) == 0:
                    atualizar_status_tarefa(tarefa_id, 'SEM_ROSTO')
                else:
                    for f_idx, face in enumerate(faces, start=1):
                        bbox = face.bbox.astype(int)
                        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                        x2, y2 = min(img_bgr.shape[1], bbox[2]), min(img_bgr.shape[0], bbox[3])
                        alt = y2 - y1
                        larg = x2 - x1
                        prop = larg / float(alt) if alt > 0 else 0
                        
                        score = face.det_score if hasattr(face, 'det_score') else 0.0
                        rosto_recortado = img_bgr[y1:y2, x1:x2]

                        if alt < tamanho_min_face or larg < tamanho_min_face or prop < 0.55 or prop > 1.35:
                            if rosto_recortado.size > 0:
                                nome_debug = f"lixo_score{score:.2f}_alt{alt}_larg{larg}_{hash_base[:6]}.jpg"
                                cv2.imwrite(os.path.join(pasta_rejeitados, nome_debug), rosto_recortado)
                            continue

                        vetor_512d = face.embedding.astype(np.float32)
                        hash_face = hashlib.sha256(f"{hash_base}_face_{f_idx}".encode()).hexdigest()
                        caminho_crop_rel = f"crop_virtual_{id_extracao}_{f_idx}.jpg"

                        status, ind_id, historico_lotes = registrar_face(
                            vetor_512d=vetor_512d,
                            hash_foto=hash_face,
                            id_extracao=id_extracao,
                            caminho_original=caminho_relativo,
                            caminho_crop=caminho_crop_rel,
                            limite_distancia=tolerancia_match
                        )

                        pasta_suspeito = os.path.join(pasta_destino, str(ind_id))
                        os.makedirs(pasta_suspeito, exist_ok=True)
                        nome_crop = f"celular_{id_extracao}_face_{f_idx}.jpg"
                        caminho_final_crop = os.path.join(pasta_suspeito, nome_crop)

                        if rosto_recortado.size > 0:
                            cv2.imwrite(caminho_final_crop, rosto_recortado)
                        
                        st.session_state["ultimos_crops"].insert(0, {
                            "caminho": caminho_final_crop,
                            "id": str(ind_id),
                            "status": status,
                            "lote": id_extracao
                        })
                        if len(st.session_state["ultimos_crops"]) > 12:
                            st.session_state["ultimos_crops"].pop()

                atualizar_status_tarefa(tarefa_id, 'CONCLUIDO')

                stats_live = obter_estatisticas_sistema()
                f_st = stats_live["fila"]
                t_fil = sum(f_st.values())
                c_fil = f_st.get("CONCLUIDO", 0) + f_st.get("SEM_ROSTO", 0) + f_st.get("FALHA", 0)
                p_pct = (c_fil / t_fil) if t_fil > 0 else 0.0
                prog_bar.progress(p_pct, text=f"Progresso de Processamento: {p_pct*100:.1f}% ({c_fil}/{t_fil})")

                with gallery_placeholder.container():
                    st.caption("Recortes Recentes:")
                    if st.session_state["ultimos_crops"]:
                        cols_g = st.columns(min(len(st.session_state["ultimos_crops"]), 6))
                        for idx_g, item_g in enumerate(st.session_state["ultimos_crops"][:6]):
                            with cols_g[idx_g % 6]:
                                if os.path.exists(item_g["caminho"]):
                                    st.image(item_g["caminho"], caption=f"{item_g['status']} | {item_g['lote']}", width=110)

        if not st.session_state["worker_rodando"] and st.session_state["ultimos_crops"]:
            st.caption("Histórico de Recortes:")
            cols_g = st.columns(min(len(st.session_state["ultimos_crops"]), 6))
            for idx_g, item_g in enumerate(st.session_state["ultimos_crops"][:6]):
                with cols_g[idx_g % 6]:
                    if os.path.exists(item_g["caminho"]):
                        st.image(item_g["caminho"], caption=f"{item_g['status']} | {item_g['lote']}", width=110)

    with aba2:
        st.subheader("Auditoria de Identidades e Reincidências")
        
        pasta_crops_audit = st.text_input("Diretório de Armazenamento de Identidades:", value=caminho_padrao_destino, key="path_crops_audit")
        
        if not os.path.exists(pasta_crops_audit):
            st.warning(f"Diretório '{pasta_crops_audit}' inacessível.")
        else:
            ids_suspeitos = [d for d in os.listdir(pasta_crops_audit) if os.path.isdir(os.path.join(pasta_crops_audit, d))]
            
            total_individuos = len(ids_suspeitos)
            total_crops = 0
            reincidentes_count = 0
            lotes_mapeados = set()
            dados_suspeitos = []

            for ind_id in ids_suspeitos:
                caminho_id = os.path.join(pasta_crops_audit, ind_id)
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
                    "id": ind_id,
                    "caminho": caminho_id,
                    "fotos": fotos,
                    "qtd_fotos": qtd_fotos,
                    "lotes": list(lotes_do_suspeito),
                    "reincidente": eh_reincidente
                })

            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            with col_kpi1:
                st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{total_individuos}</p><p class="metric-lbl">Total de Cadastros</p></div>', unsafe_allow_html=True)
            with col_kpi2:
                st.markdown(f'<div class="metric-card-alert"><p class="metric-val" style="color:#ef4444;">{reincidentes_count}</p><p class="metric-lbl">Reincidentes Mapeados</p></div>', unsafe_allow_html=True)
            with col_kpi3:
                st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#10b981;">{total_crops}</p><p class="metric-lbl">Evidências Salvas</p></div>', unsafe_allow_html=True)
            with col_kpi4:
                st.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#f59e0b;">{len(lotes_mapeados)}</p><p class="metric-lbl">Lotes Analisados</p></div>', unsafe_allow_html=True)

            st.write("")
            
            col_filtro1, col_filtro2 = st.columns([1.2, 2])
            with col_filtro1:
                filtro_modo = st.radio("Critério de Exibição:", ["Apenas Reincidentes", "Todos os Registros"], horizontal=True, key="filtro_auditoria")
            with col_filtro2:
                termo_busca = st.text_input("Filtrar por Identificador ou Lote:", value="", placeholder="Ex: celular_01...", key="busca_auditoria").strip().lower()

            st.write("")

            suspeitos_exibir = []
            for item in dados_suspeitos:
                if filtro_modo == "Apenas Reincidentes" and not item["reincidente"]:
                    continue
                if termo_busca:
                    match_id = termo_busca in item["id"].lower()
                    match_lote = any(termo_busca in l.lower() for l in item["lotes"])
                    if not (match_id or match_lote):
                        continue
                suspeitos_exibir.append(item)

            if not suspeitos_exibir:
                st.info("Nenhum dado localizado para os critérios informados.")
            else:
                st.caption(f"Retornando {len(suspeitos_exibir)} registro(s).")

                for item in suspeitos_exibir:
                    status_badge = '<span class="badge-reincidente">REINCIDENTE</span>' if item["reincidente"] else '<span class="badge-lote">REGISTRO ÚNICO</span>'
                    lotes_str = ", ".join(item["lotes"])
                    
                    header_card = f"Identificação: {item['id']} | {item['qtd_fotos']} evidência(s) em {len(item['lotes'])} lote(s)"
                    
                    with st.expander(header_card, expanded=item["reincidente"]):
                        st.markdown(f"{status_badge} &nbsp; <span style='color:#94a3b8; font-size:0.85rem;'><b>Origem:</b> {lotes_str}</span>", unsafe_allow_html=True)
                        st.write("")
                        
                        cols = st.columns(min(len(item["fotos"]), 7))
                        for idx, foto_nome in enumerate(item["fotos"]):
                            caminho_foto = os.path.join(item["caminho"], foto_nome)
                            
                            lote_origem = "Geral"
                            if "celular_" in foto_nome:
                                partes = foto_nome.split("_")
                                if len(partes) > 1:
                                    lote_origem = partes[1]
                                    
                            with cols[idx % 7]:
                                if os.path.exists(caminho_foto):
                                    st.image(caminho_foto, caption=f"{lote_origem}", width=120)

    with aba3:
        st.subheader("Busca de Identidades por Referência")

        c1, c2 = st.columns([1.2, 1])
        
        caminho_padrao_ref = r"C:\Users\saulo.borges\Downloads\facial_teste\referencia"
        
        with c1:
            opcao_origem = st.radio("Método de Entrada:", ["Diretório Local", "Upload Manual"], horizontal=True, key="origem_ref")
            
            caminho_imagem_analisar = None
            imagem_bytes_upload = None

            if opcao_origem == "Diretório Local":
                pasta_ref_input = st.text_input("Caminho do Diretório:", value=caminho_padrao_ref, key="path_ref")
                if os.path.exists(pasta_ref_input):
                    fotos_ref = [f for f in os.listdir(pasta_ref_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if fotos_ref:
                        foto_selecionada = st.selectbox("Selecione o arquivo:", fotos_ref, key="sel_ref")
                        caminho_imagem_analisar = os.path.join(pasta_ref_input, foto_selecionada)
                    else:
                        st.warning("Diretório vazio ou sem imagens compatíveis.")
                else:
                    st.error("Caminho especificado inválido.")
            else:
                uploaded_file = st.file_uploader("Selecionar Arquivo:", type=["jpg", "jpeg", "png"], key="up_ref")
                if uploaded_file is not None:
                    imagem_bytes_upload = uploaded_file.read()

        with c2:
            st.caption("Visualização:")
            img_bgr = None
            
            if caminho_imagem_analisar and os.path.exists(caminho_imagem_analisar):
                img_bgr = cv2.imread(caminho_imagem_analisar)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=180)
            elif imagem_bytes_upload is not None:
                nparr = np.frombuffer(imagem_bytes_upload, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=180)

        st.write("")

        if st.button("Executar Pesquisa Vetorial", type="primary", key="btn_ref"):
            if img_bgr is None:
                st.error("Forneça uma imagem válida para iniciar a varredura.")
            else:
                progress_bar = st.progress(0, text="Iniciando plataforma...")
                status_box = st.status("Processando biometria...", expanded=True)
                
                status_box.write("Carregando arquitetura neural...")
                progress_bar.progress(25, text="Carregando modelo...")
                app_arcface = carregar_modelo_arcface(confianca_minima)
                
                if app_arcface is None:
                    status_box.update(label="Falha de inicialização.", state="error")
                    st.error("Erro interno ao carregar dependências biométricas.")
                else:
                    status_box.write("Extraindo features em alta resolução...")
                    progress_bar.progress(50, text="Localizando faces...")
                    
                    faces_raw = app_arcface.get(img_bgr)

                    faces_validas = []
                    for f in faces_raw:
                        bbox = f.bbox.astype(int)
                        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                        x2, y2 = min(img_bgr.shape[1], bbox[2]), min(img_bgr.shape[0], bbox[3])
                        alt = y2 - y1
                        larg = x2 - x1
                        prop = larg / float(alt) if alt > 0 else 0
                        if alt < tamanho_min_face or larg < tamanho_min_face or prop < 0.65 or prop > 1.25:
                            pass
                        else:
                            faces_validas.append(f)

                    if len(faces_validas) == 0:
                        status_box.update(label="Padrão biométrico não identificado.", state="warning")
                        progress_bar.progress(100, text="Processo finalizado sem extração.")
                        st.warning("Não foi possível isolar características faciais com o ArcFace.")
                    else:
                        status_box.write(f"Consultando banco vetorial para {len(faces_validas)} biometria(s)...")
                        progress_bar.progress(75, text="Executando query no PostgreSQL...")
                        
                        time.sleep(0.3)
                        progress_bar.progress(100, text="Concluído.")
                        status_box.update(label="Pesquisa finalizada.", state="complete")
                        
                        for idx_f, face in enumerate(faces_validas, start=1):
                            vetor_512d = face.embedding.astype(np.float32)
                            resultados = buscar_historico_completo_por_vetor(vetor_512d, limite_distancia=tolerancia_match)
                            
                            st.write(f"#### Perfil Identificado #{idx_f}")
                            
                            if not resultados:
                                st.info("Sem correspondência histórica. Registro inédito no sistema.")
                            else:
                                for res in resultados:
                                    ind_id = res["individuo_id"]
                                    dist = res["distancia"]
                                    ocorrencias = res["ocorrencias"]
                                    
                                    lotes_unicos = sorted(list(set(oc["lote"] for oc in ocorrencias)))
                                    
                                    col_res1, col_res2 = st.columns([1, 2.2])
                                    
                                    with col_res1:
                                        if dist <= 0.38:
                                            st.markdown('<span class="badge-reincidente">MATCH POSITIVO</span>', unsafe_allow_html=True)
                                        else:
                                            st.markdown('<span class="badge-lote">CORRESPONDÊNCIA PARCIAL</span>', unsafe_allow_html=True)
                                        
                                        st.write("")
                                        st.markdown(f"**Identificador:** `{ind_id}`")
                                        st.markdown(f"**Proximidade:** `{dist:.4f}`")
                                        st.markdown(f"**Frequência:** `{len(ocorrencias)}`")
                                        st.markdown(f"**Origens:** `{', '.join(lotes_unicos)}`")
                                        
                                    with col_res2:
                                        st.caption("Detalhamento de Ocorrências:")
                                        df_oc = pd.DataFrame(ocorrencias)
                                        if not df_oc.empty:
                                            df_oc_display = df_oc[['lote', 'caminho_arquivo', 'data_ocorrencia']].rename(columns={
                                                'lote': 'Lote',
                                                'caminho_arquivo': 'Arquivo Matriz',
                                                'data_ocorrencia': 'Registro'
                                            })
                                            st.dataframe(df_oc_display, use_container_width=True)

                                    st.caption("Acervo Biométrico Mapeado (Disco Físico):")
                                    
                                    pasta_fisica_id = os.path.join(caminho_padrao_destino, str(ind_id))
                                    if os.path.exists(pasta_fisica_id):
                                        fotos_reais = [f for f in os.listdir(pasta_fisica_id) if f.lower().endswith(('.jpg', '.png'))]
                                        
                                        if fotos_reais:
                                            cols_oc = st.columns(min(len(fotos_reais), 7))
                                            for idx_oc, foto_nome in enumerate(fotos_reais):
                                                caminho_crop_full = os.path.join(pasta_fisica_id, foto_nome)
                                                
                                                lote_origem = "Geral"
                                                if "celular_" in foto_nome:
                                                    partes = foto_nome.split("_")
                                                    if len(partes) > 1:
                                                        lote_origem = partes[1]
                                                        
                                                with cols_oc[idx_oc % 7]:
                                                    st.image(caminho_crop_full, caption=f"{lote_origem}", width=110)
                                        else:
                                            st.caption("Pasta física encontrada, mas está vazia.")
                                    else:
                                        st.warning("Imagens físicas ausentes ou deletadas do diretório.")

if __name__ == "__main__":
    carregar_painel()