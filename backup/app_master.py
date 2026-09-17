import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="DASHBOARD SISTEMA DE VISUALIZAÇÃO | SESP-PR",
    # page_icon="",
    # layout="wide",
    # initial_sidebar_state="expanded"
)

# Estilos CSS customizados para métricas e cards
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .metric-card { background-color: #1e2130; padding: 20px; border-radius: 10px; border-left: 5px solid #00f2fe; margin-bottom: 20px; }
    .outlier-card { background-color: #1e2130; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DADOS REAIS DO BENCHMARK (HP ProBook)
# ==========================================
# Dados de Benchmark de Hardware (Módulos 1 e 2)
df_hw = pd.DataFrame({
    'Modelo': ['MTCNN', 'YOLOv8-Face', 'SSD (Caffe)'],
    'Latência (ms)': [1494.90, 86.36, 43.11],
    'RAM Alocada (MB)': [57.20, 3.10, 1.47],
    'Uso CPU (ms)': [1731.90, 561.90, 42.40]
})

# Dados de Clusterização DBSCAN para o Gráfico de Dispersão
# Simulando a redução de dimensionalidade (MDS) de 512D para 3D para visualização
np.random.seed(42)
n_inliers = 1843
n_outliers = 32

df_clusters = pd.DataFrame({
    'X': np.concatenate([np.random.normal(0, 1, n_inliers), np.random.normal(4, 1.5, n_outliers)]),
    'Y': np.concatenate([np.random.normal(0, 1, n_inliers), np.random.normal(4, 1.5, n_outliers)]),
    'Z': np.concatenate([np.random.normal(0, 1, n_inliers), np.random.normal(4, 1.5, n_outliers)]),
    'Status': ['Indivíduo Único'] * n_inliers + ['Outlier (Anomalia)'] * n_outliers,
    'Tamanho': [5] * n_inliers + [12] * n_outliers
})

# ==========================================
# 3. BARRA LATERAL (MENU DE NAVEGAÇÃO)
# ==========================================
#st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Bras%C3%A3o_do_Paran%C3%A1.svg/1200px-Bras%C3%A3o_do_Paran%C3%A1.svg.png", width=100)
st.sidebar.title("Sistema de Visualização - SESP-PR")
st.sidebar.caption("Sistema de Monitoramento e Análise de Triagem Biométrica")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navegação do Sistema",
    ["1. Visão Geral (KPIs)", "2. Auditoria de Hardware", "3. Inteligência Biométrica", "4. Arquitetura C4"]
)

st.sidebar.markdown("---")
st.sidebar.info("Status: Operação Assíncrona Ativa")
st.sidebar.success("Motor: YOLOv8 + ArcFace + DBSCAN")

# ==========================================
# 4. RENDERIZAÇÃO DAS PÁGINAS
# ==========================================

# ------------------------------------------
# PÁGINA 1: VISÃO GERAL
# ------------------------------------------
if menu == "1. Visão Geral (KPIs)":
    st.title(" Visão Geral do Sistema")
    st.markdown("Monitoramento de triagem e funil de processamento biométrico.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Capturas Brutas", "7.478", delta="Ingestão I/O")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Faces Válidas (ArcFace)", "1.875", delta="-5.603 Filtro de Qualidade", delta_color="normal")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Indivíduos Únicos", "1.843", delta="DBSCAN Clustering")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="outlier-card">', unsafe_allow_html=True)
        st.metric("Outliers Detectados", "32", delta="Alerta Operacional", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Funil de Retenção Biométrica")
    
    # Funil Plotly
    funnel_data = dict(
        number=[7478, 3104, 1875, 1843],
        stage=["Imagens Capturadas", "Faces Detectadas (YOLOv8)", "Embeddings 512D (ArcFace)", "Identidades Únicas (DBSCAN)"]
    )
    fig_funnel = px.funnel(funnel_data, x='number', y='stage', color_discrete_sequence=['#00f2fe'])
    fig_funnel.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_funnel)

# ------------------------------------------
# PÁGINA 2: AUDITORIA DE HARDWARE
# ------------------------------------------
elif menu == "2. Auditoria de Hardware":
    st.title(" Auditoria de Hardware (Edge Computing)")
    st.markdown("Avaliação de viabilidade computacional para dispositivos de borda com recursos limitados.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Latência de Inferência")
        fig_lat = px.bar(df_hw, x='Modelo', y='Latência (ms)', color='Modelo', text_auto=True,
                         color_discrete_sequence=['#ff4b4b', '#00f2fe', '#00ff87'])
        fig_lat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_lat)
        
    with col2:
        st.subheader(" Consumo de Memória (RAM)")
        fig_ram = px.bar(df_hw, x='Modelo', y='RAM Alocada (MB)', color='Modelo', text_auto=True,
                         color_discrete_sequence=['#ff4b4b', '#00f2fe', '#00ff87'])
        fig_ram.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ram)
        
    st.info("**Veredito Arquitetural:** O modelo YOLOv8-Face apresentou o melhor equilíbrio, garantindo precisão semântica avançada com latência aceitável e alocação de memória controlada, sendo eleito para o Módulo de Detecção.")

# ------------------------------------------
# PÁGINA 3: INTELIGÊNCIA BIOMÉTRICA
# ------------------------------------------
elif menu == "3. Inteligência Biométrica":
    st.title(" Agrupamento Espacial e Outliers")
    st.markdown("Projeção tridimensional dos vetores matemáticos extraídos pelo ArcFace e agrupados pelo DBSCAN.")
    
    st.markdown('<div class="outlier-card">Foram isoladas 32 assinaturas vetoriais com alta similaridade (Distância Euclidiana), indicando multiplicidade de registros para a mesma identidade física (Outliers).</div>', unsafe_allow_html=True)
    
    # Gráfico 3D Scatter (Simulando o MDS do Orange Canvas)
    fig_3d = px.scatter_3d(df_clusters, x='X', y='Y', z='Z',
                           color='Status',
                           size='Tamanho',
                           color_discrete_map={'Indivíduo Único': '#00f2fe', 'Outlier (Anomalia)': '#ff4b4b'},
                           opacity=0.7,
                           title="Mapeamento Topológico de Identidades (MDS de 512D para 3D)")
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_3d, height=700)

# ------------------------------------------
# PÁGINA 4: ARQUITETURA C4
# ------------------------------------------
elif menu == "4. Arquitetura C4":
    st.title("Arquitetura de Sistemas (Modelo C4)")
    st.markdown("Especificações de engenharia de software da plataforma ID-Hunter.")
    
    st.subheader("Nível 1: Diagrama de Contexto")
    st.info("O ID-Hunter atua como núcleo inteligente, consumindo imagens brutas de CFTV passivamente. O Operador acessa via navegador web para monitorar KPIs e interagir com o dashboard sem lidar com a complexidade subjacente.")
    
    st.subheader("Nível 2: Diagrama de Contêiner (Metodologia)")
    
    # Tabela descritiva dos containers
    df_containers = pd.DataFrame({
        "Contêiner": ["Módulo 1 & 2: Motor de Detecção", "Módulo 3: Inteligência Biométrica", "Módulo 4: Dashboard Analytics", "Armazenamento Local (Edge)"],
        "Tecnologias": ["Python, YOLOv8", "Python, ArcFace, DBSCAN", "Python, Streamlit, Plotly", "File System, CSV"],
        "Responsabilidade": ["Filtragem de imagens descartáveis e extração de bounding boxes.", "Vetorização (512 dimensões) e agrupamento de outliers via distância euclidiana.", "Interface web interativa para visualização de resultados e métricas de HW.", "Persistência assíncrona de dados limpos e metadados na borda."]
    })
    
    st.table(df_containers)