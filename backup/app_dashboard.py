import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E ESTILOS
# ==========================================
st.set_page_config(page_title="ID-Hunter Dashboard", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #deff9a; }
    </style>
    """, unsafe_allow_html=True)

st.title("HUNTER CLEAN DASHBOARD")
st.markdown("---")

# ==========================================
# CAMADA DE DADOS
# ==========================================
METRICS_PATH = "resultados_id_hunter/metricas_extracao.csv"
VETORES_PATH = "resultados_id_hunter/identificacao/vetores_512d.csv"

# ==========================================
# SIDEBAR - FILTROS E STATUS
# ==========================================
st.sidebar.header("Configurações")
pipeline_status = st.sidebar.status("Pipeline Ativo", state="complete")
st.sidebar.info("Sistema operando em modo: Investigação Forense")

# ==========================================
# 1. KPIs PRINCIPAIS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Capturas", "7.478", delta="Ingestão")
with col2:
    st.metric("Duplicados Barrados", "3.969", delta="-53%", delta_color="inverse")
with col3:
    st.metric("Pessoas Únicas", "1.843", delta="DBSCAN")
with col4:
    # Corrigido para 'inverse' e terminologia ajustada para Outliers
    st.metric("Outliers Detectados", "32", delta="Alerta", delta_color="inverse")

st.markdown("###")

# ==========================================
# 2. ARENA DE BENCHMARKING (GRÁFICOS)
# ==========================================
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(" Arena de Extração (Latência)")
    # Dados reais que gerámos no Módulo 2
    data_bench = {
        'Algoritmo': ['MTCNN', 'YOLOv8-Face', 'SSD (Caffe)'],
        'Latência (ms)': [609.94, 51.09, 32.66]
    }
    df_bench = pd.DataFrame(data_bench)
    fig_bench = px.bar(df_bench, x='Algoritmo', y='Latência (ms)', color='Algoritmo',
                       color_discrete_sequence=['#ff4b4b', '#deff9a', '#daffde'],
                       text_auto=True)
    st.plotly_chart(fig_bench, use_container_width=True)

with right_col:
    st.subheader(" Composição do Dataset")
    # Dados reais da nossa triagem semântica
    data_comp = {
        'Categoria': ['Faces Limpas', 'Oclusões', 'Duplicados'],
        'Quantidade': [3104, 405, 3969]
    }
    fig_comp = px.pie(data_comp, values='Quantidade', names='Categoria', hole=0.5,
                      color_discrete_sequence=['#deff9a', '#daffde', '#333'])
    st.plotly_chart(fig_comp, use_container_width=True)

# ==========================================
# 3. GALERIA DE RESULTADOS BIOMÉTRICOS
# ==========================================
st.markdown("---")
st.subheader("Detecção de Identidades Duplicadas (Outliers)")

if os.path.exists(VETORES_PATH):
    df_ids = pd.read_csv(VETORES_PATH)
    
    # Simulação da exibição das identidades geradas pelo DBSCAN
    st.write("Amostra da base de dados com as respetivas assinaturas matemáticas agrupadas:")
    st.dataframe(df_ids.head(10), use_container_width=True)
else:
    st.warning("Arquivo de vetores não encontrado. Certifique-se de executar o Módulo 3.")

# ==========================================
# RODAPÉ
# ==========================================
st.markdown("---")
st.caption("ID-Hunter v2.0 | Desenvolvido para SESP-PR | UTFPR")