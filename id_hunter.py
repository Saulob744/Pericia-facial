import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import time
import hashlib
import warnings
import shutil
from database_manager import (
    enfileirar_tarefa, pegar_tarefa_pendente, atualizar_status_tarefa,
    registrar_face, obter_estatisticas_sistema, buscar_historico_completo_por_vetor,
    resetar_banco_dados
)

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(layout="wide", page_title="ID-Hunter | Central Unificada", page_icon="🔬")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    p, span, label, div { color: #cbd5e1; }
    .stMarkdown p { color: #cbd5e1 !important; font-size: 0.95rem; }
    .pericia-header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px 32px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
    .pericia-title { color: #f8fafc !important; font-size: 1.75rem; font-weight: 600; margin: 0 0 8px 0; letter-spacing: -0.025em; }
    .pericia-subtitle { color: #94a3b8 !important; font-size: 0.95rem; margin: 0; }
    .metric-card-clean, .metric-card-alert { background-color: #1e293b; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border: 1px solid #334155; }
    .metric-card-clean { border-top: 3px solid #3b82f6; }
    .metric-card-alert { border-top: 3px solid #ef4444; }
    .metric-val { font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1.2; }
    .metric-lbl { font-size: 0.75rem; color: #94a3b8 !important; margin: 8px 0 0 0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .badge-lote, .badge-reincidente, .badge-match { font-weight: 600; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; letter-spacing: 0.025em; }
    .badge-lote { background-color: #3b82f6; color: #ffffff !important; }
    .badge-reincidente { background-color: #ef4444; color: #ffffff !important; }
    .badge-match { background-color: #10b981; color: #ffffff !important; }
    .stImage img { border-radius: 8px; border: 1px solid #334155; object-fit: cover; max-height: 150px; transition: all 0.2s ease-in-out; }
    .stImage img:hover { transform: scale(1.02); border-color: #3b82f6; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    </style>
""", unsafe_allow_html=True)

if "worker_rodando" not in st.session_state: st.session_state["worker_rodando"] = False
if "parar_solicitado" not in st.session_state: st.session_state["parar_solicitado"] = False
if "ultimos_crops" not in st.session_state: st.session_state["ultimos_crops"] = []

@st.cache_resource
def carregar_modelo_arcface():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.20)
        return app
    except Exception as e:
        st.error(f"Erro: {e}")
        return None

def calcular_sha256(caminho_arquivo: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(caminho_arquivo, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

def carregar_painel():
    st.markdown('<div class="pericia-header"><h1 class="pericia-title">ID-Hunter | Central Unificada</h1><p class="pericia-subtitle">Sistema de varredura automatizada, extração biométrica e busca.</p></div>', unsafe_allow_html=True)

    aba1, aba2, aba3 = st.tabs(["🔬 Processamento e Fila", "📂 Auditoria e Reincidência", "🔎 Busca Biométrica"])
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho_padrao_origem = os.path.join(BASE_DIR, "entradas_celular")
    caminho_padrao_destino = os.path.join(BASE_DIR, "armazenamento_crops")
    caminho_padrao_ref = os.path.join(BASE_DIR, "referencia")
    caminho_padrao_ignoradas = os.path.join(BASE_DIR, "armazenamento_ignoradas")

    os.makedirs(caminho_padrao_destino, exist_ok=True)
    os.makedirs(caminho_padrao_ignoradas, exist_ok=True)

    with aba1:
        st.subheader("Processamento de Evidências")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1: pasta_origem = st.text_input("Diretório Origem:", value=caminho_padrao_origem)
        with col_p2: pasta_destino = st.text_input("Diretório Destino:", value=caminho_padrao_destino)
        with col_p3: pasta_ignoradas = st.text_input("Caixa Preta (Ignoradas):", value=caminho_padrao_ignoradas)

        col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1.2, 1.2])
        with col_btn1:
            if st.button("📋 Alimentar Fila", use_container_width=True):
                if not os.path.exists(pasta_origem):
                    st.error("Diretório não localizado.")
                else:
                    lotes = [d for d in os.listdir(pasta_origem) if os.path.isdir(os.path.join(pasta_origem, d))]
                    if not lotes: st.warning("Nenhum lote localizado.")
                    else:
                        total = 0
                        with st.spinner("Indexando..."):
                            for lote in lotes:
                                caminho_lote = os.path.join(pasta_origem, lote)
                                for root, _, files in os.walk(caminho_lote):
                                    for file in files:
                                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                            caminho_abs = os.path.join(root, file)
                                            caminho_rel = os.path.relpath(caminho_abs, pasta_origem)
                                            enfileirar_tarefa(lote, caminho_rel)
                                            total += 1
                        st.success(f"{total} inseridos na fila.")

        with col_btn2:
            if st.button("▶️ Iniciar Motor de Extração", type="primary", use_container_width=True, disabled=st.session_state["worker_rodando"]):
                st.session_state["worker_rodando"] = True
                st.session_state["parar_solicitado"] = False
                st.rerun()

        with col_btn3:
            if st.button("⏹️ Interromper Processamento", use_container_width=True, disabled=not st.session_state["worker_rodando"]):
                st.session_state["parar_solicitado"] = True
                st.warning("Interrupção agendada.")

        with st.expander("⚠️ Reset de Banco de Dados"):
            confirmar = st.checkbox("Estou ciente.")
            if st.button("Excluir Banco", type="primary"):
                if confirmar:
                    resetar_banco_dados()
                    st.session_state["ultimos_crops"] = []
                    st.success("Banco formatado.")
                    time.sleep(1)
                    st.rerun()
                else: st.error("Confirme a operação.")

        st.divider()
        stats = obter_estatisticas_sistema()
        total_fila = sum(stats["fila"].values())
        concluidas = stats["fila"].get("CONCLUIDO", 0) + stats["fila"].get("SEM_ROSTO", 0) + stats["fila"].get("FALHA", 0)
        restantes = total_fila - concluidas
        prog_pct = (concluidas / total_fila) if total_fila > 0 else 0.0

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        col_s1.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{stats["fila"].get("PENDENTE", 0)}</p><p class="metric-lbl">Pendentes</p></div>', unsafe_allow_html=True)
        col_s2.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#10b981;">{stats["fila"].get("CONCLUIDO", 0)}</p><p class="metric-lbl">Extraídas</p></div>', unsafe_allow_html=True)
        col_s3.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#f59e0b;">{stats["fila"].get("SEM_ROSTO", 0)}</p><p class="metric-lbl">Sem Face</p></div>', unsafe_allow_html=True)
        col_s4.markdown(f'<div class="metric-card-alert"><p class="metric-val" style="color:#ef4444;">{stats["biometria"].get("total_individuos", 0)}</p><p class="metric-lbl">Únicos</p></div>', unsafe_allow_html=True)
        col_s5.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{stats["biometria"].get("total_ocorrencias", 0)}</p><p class="metric-lbl">Ocorrências</p></div>', unsafe_allow_html=True)

        if st.session_state["worker_rodando"]:
            prog_bar = st.progress(prog_pct, text=f"Progresso: {prog_pct*100:.1f}% | Restantes: {restantes} de {total_fila}")
            status_live = st.status("Processando...", expanded=True)
            gallery = st.empty()
            app_arcface = carregar_modelo_arcface()

            while st.session_state["worker_rodando"]:
                if st.session_state["parar_solicitado"]:
                    status_live.update(label="Interrompido.", state="error")
                    st.session_state["worker_rodando"] = False
                    st.session_state["parar_solicitado"] = False
                    st.rerun()
                    break

                tarefa = pegar_tarefa_pendente()
                if not tarefa:
                    status_live.update(label="Fila esgotada.", state="complete")
                    st.session_state["worker_rodando"] = False
                    break

                t_id, ext_id, cam_rel = tarefa
                cam_fis = os.path.join(pasta_origem, cam_rel)
                status_live.write(f"Arquivo: {cam_rel}")

                h_base = calcular_sha256(cam_fis)
                if not h_base or not os.path.exists(cam_fis):
                    atualizar_status_tarefa(t_id, 'FALHA')
                    time.sleep(0.05)
                    continue

                img = cv2.imread(cam_fis)
                if img is None:
                    atualizar_status_tarefa(t_id, 'FALHA')
                    try:
                        shutil.copy(cam_fis, os.path.join(pasta_ignoradas, f"falha_{os.path.basename(cam_fis)}"))
                    except: pass
                    time.sleep(0.05)
                    continue

                faces = app_arcface.get(img)
                if not faces:
                    atualizar_status_tarefa(t_id, 'SEM_ROSTO')
                    cv2.imwrite(os.path.join(pasta_ignoradas, f"semrosto_{ext_id}_{os.path.basename(cam_fis)}"), img)
                else:
                    rosto_salvo = False
                    for idx, f in enumerate(faces, 1):
                        bx = f.bbox.astype(int)
                        x1, y1, x2, y2 = max(0, bx[0]), max(0, bx[1]), min(img.shape[1], bx[2]), min(img.shape[0], bx[3])
                        alt_rosto, larg_rosto = y2 - y1, x2 - x1
                        
                        if alt_rosto < 20 or larg_rosto < 20: continue

                        rosto_salvo = True
                        margem_y, margem_x = int(alt_rosto * 0.25), int(larg_rosto * 0.25)
                        x1_n, y1_n = max(0, x1 - margem_x), max(0, y1 - margem_y)
                        x2_n, y2_n = min(img.shape[1], x2 + margem_x), min(img.shape[0], y2 + margem_y)

                        crop = img[y1_n:y2_n, x1_n:x2_n]
                        vec = f.embedding.astype(np.float32)
                        h_face = hashlib.sha256(f"{h_base}_f{idx}".encode()).hexdigest()
                        c_crop = f"crop_{ext_id}_{idx}.jpg"
                        
                        stt, ind_id, _ = registrar_face(vec, h_face, ext_id, cam_rel, c_crop)
                        
                        p_susp = os.path.join(pasta_destino, str(ind_id))
                        os.makedirs(p_susp, exist_ok=True)
                        p_final = os.path.join(p_susp, f"celular_{ext_id}_f{idx}.jpg")
                        
                        if crop.size > 0: cv2.imwrite(p_final, crop)
                        
                        st.session_state["ultimos_crops"].insert(0, {"caminho": p_final, "status": stt, "lote": ext_id})
                        if len(st.session_state["ultimos_crops"]) > 12: st.session_state["ultimos_crops"].pop()
                    
                    if rosto_salvo:
                        atualizar_status_tarefa(t_id, 'CONCLUIDO')
                    else:
                        atualizar_status_tarefa(t_id, 'SEM_ROSTO')
                        cv2.imwrite(os.path.join(pasta_ignoradas, f"rejeitada_filtro_{ext_id}_{os.path.basename(cam_fis)}"), img)

                stats_live = obter_estatisticas_sistema()
                t_fil = sum(stats_live["fila"].values())
                c_fil = stats_live["fila"].get("CONCLUIDO", 0) + stats_live["fila"].get("SEM_ROSTO", 0) + stats_live["fila"].get("FALHA", 0)
                rest = t_fil - c_fil
                p_pct = (c_fil / t_fil) if t_fil > 0 else 0.0
                prog_bar.progress(p_pct, text=f"Progresso: {p_pct*100:.1f}% | Restantes: {rest} de {t_fil}")
                
                with gallery.container():
                    if st.session_state["ultimos_crops"]:
                        cg = st.columns(6)
                        for i, it in enumerate(st.session_state["ultimos_crops"][:6]):
                            if os.path.exists(it["caminho"]): cg[i%6].image(it["caminho"], width=110)
                
                time.sleep(0.05)

        if not st.session_state["worker_rodando"] and st.session_state["ultimos_crops"]:
            st.caption("Últimos Recortes:")
            cg = st.columns(6)
            for i, it in enumerate(st.session_state["ultimos_crops"][:6]):
                if os.path.exists(it["caminho"]): cg[i%6].image(it["caminho"], width=110)

    with aba2:
        st.subheader("Auditoria de Identidades")
        pasta_audit = st.text_input("Armazenamento:", value=pasta_destino)
        if os.path.exists(pasta_audit):
            ids = [d for d in os.listdir(pasta_audit) if os.path.isdir(os.path.join(pasta_audit, d))]
            reinc, tot_crops, lotes_map, dados = 0, 0, set(), []
            for i in ids:
                cid = os.path.join(pasta_audit, i)
                fts = [f for f in os.listdir(cid) if f.lower().endswith(('.jpg', '.png'))]
                q = len(fts)
                tot_crops += q
                lts = set(f.split("_")[1] if "celular_" in f and len(f.split("_"))>1 else "Geral" for f in fts)
                lotes_map.update(lts)
                e_reinc = q > 1 or len(lts) > 1
                if e_reinc: reinc += 1
                dados.append({"id": i, "caminho": cid, "fotos": fts, "qtd": q, "lotes": list(lts), "reincidente": e_reinc})

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#3b82f6;">{len(ids)}</p><p class="metric-lbl">Cadastros</p></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="metric-card-alert"><p class="metric-val" style="color:#ef4444;">{reinc}</p><p class="metric-lbl">Reincidentes</p></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#10b981;">{tot_crops}</p><p class="metric-lbl">Evidências</p></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="metric-card-clean"><p class="metric-val" style="color:#f59e0b;">{len(lotes_map)}</p><p class="metric-lbl">Lotes</p></div>', unsafe_allow_html=True)

            cf1, cf2 = st.columns([1.2, 2])
            filtro = cf1.radio("Exibir:", ["Apenas Reincidentes", "Todos"])
            busca = cf2.text_input("Filtrar:").strip().lower()

            exibir = [d for d in dados if not (filtro == "Apenas Reincidentes" and not d["reincidente"]) and (not busca or busca in d["id"].lower() or any(busca in l.lower() for l in d["lotes"]))]
            
            for d in exibir:
                bdg = '<span class="badge-reincidente">REINCIDENTE</span>' if d["reincidente"] else '<span class="badge-lote">ÚNICO</span>'
                with st.expander(f"{d['id']} | {d['qtd']} evidência(s)", expanded=d["reincidente"]):
                    st.markdown(f"{bdg} Origens: {', '.join(d['lotes'])}", unsafe_allow_html=True)
                    cc = st.columns(7)
                    for idx, ft in enumerate(d["fotos"]):
                        c_ft = os.path.join(d["caminho"], ft)
                        if os.path.exists(c_ft): cc[idx%7].image(c_ft, width=120)

    with aba3:
        st.subheader("Busca por Referência")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            opc = st.radio("Entrada:", ["Diretório Local", "Upload"])
            cam_img, b_up = None, None
            if opc == "Diretório Local":
                p_ref = st.text_input("Pasta:", value=caminho_padrao_ref)
                if os.path.exists(p_ref):
                    fs = [f for f in os.listdir(p_ref) if f.lower().endswith(('.jpg', '.png'))]
                    if fs: cam_img = os.path.join(p_ref, st.selectbox("Arquivo:", fs))
            else:
                up = st.file_uploader("Arquivo:")
                if up: b_up = up.read()
        with c2:
            img_bgr = None
            if cam_img and os.path.exists(cam_img):
                img_bgr = cv2.imread(cam_img)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=180)
            elif b_up:
                img_bgr = cv2.imdecode(np.frombuffer(b_up, np.uint8), cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=180)

        if st.button("Pesquisar", type="primary"):
            if img_bgr is None: st.error("Imagem inválida.")
            else:
                pb = st.progress(25)
                sb = st.status("Extraindo...", expanded=True)
                app = carregar_modelo_arcface()
                if not app: sb.update(label="Erro", state="error")
                else:
                    fs = [f for f in app.get(img_bgr) if f.bbox[3]-f.bbox[1] >= 30 and f.bbox[2]-f.bbox[0] >= 30]
                    if not fs:
                        sb.update(label="Sem face.", state="error")
                        pb.progress(100)
                    else:
                        pb.progress(75)
                        sb.update(label="Pesquisa concluída.", state="complete")
                        pb.progress(100)
                        for i, f in enumerate(fs, 1):
                            res = buscar_historico_completo_por_vetor(f.embedding.astype(np.float32), 0.42)
                            st.write(f"#### Perfil #{i}")
                            if not res: st.info("Inédito.")
                            else:
                                for r in res:
                                    c_r1, c_r2 = st.columns([1, 2.2])
                                    c_r1.markdown('<span class="badge-reincidente">MATCH</span>' if r["distancia"]<=0.38 else '<span class="badge-lote">PARCIAL</span>', unsafe_allow_html=True)
                                    c_r1.markdown(f"**ID:** `{r['individuo_id']}`<br>**Prox:** `{r['distancia']:.4f}`", unsafe_allow_html=True)
                                    df_o = pd.DataFrame(r["ocorrencias"])
                                    if not df_o.empty: c_r2.dataframe(df_o[['lote', 'caminho_arquivo', 'data_ocorrencia']])

if __name__ == "__main__":
    carregar_painel()