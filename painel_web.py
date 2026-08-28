import streamlit as st
import os

st.set_page_config(layout="wide", page_title="ID Hunter - Auditoria Visual")

def carregar_painel():
    st.title("🕵️‍♂️ ID Hunter - Auditoria de Matches (Reincidentes)")
    st.write("Verifique visualmente se a IA agrupou corretamente os rostos encontrados em diferentes pastas/celulares.")

    pasta_raiz = st.text_input("Digite o caminho raiz onde está a pasta CROPS_SUSPEITOS:", value=os.getcwd())
    pasta_crops = os.path.join("C:\\Users\\saulo.borges\\Downloads\\facial_teste\\armazenamento_crops")

    if not os.path.exists(pasta_crops):
        st.warning(f"A pasta {pasta_crops} não foi encontrada. Rode o Worker primeiro para gerar os recortes.")
        return

    st.divider()

    ids_suspeitos = os.listdir(pasta_crops)
    matches_encontrados = 0

    for individuo_id in ids_suspeitos:
        caminho_id = os.path.join(pasta_crops, individuo_id)
        
        if os.path.isdir(caminho_id):
            fotos = os.listdir(caminho_id)
            
            if len(fotos) > 1:
                matches_encontrados += 1
                st.subheader(f"👤 Suspeito ID: `{individuo_id}`")
                
                colunas = st.columns(min(len(fotos), 6))
                
                for i, foto_nome in enumerate(fotos):
                    caminho_foto = os.path.join(caminho_id, foto_nome)
                    
                    lote = "Desconhecido"
                    if "celular_" in foto_nome:
                        partes = foto_nome.split("_")
                        if len(partes) > 1:
                            lote = partes[1]
                    
                    with colunas[i % 6]:
                        st.image(caminho_foto, caption=f"Pasta: {lote}", width=150)
                
                st.divider()

    if matches_encontrados == 0:
        st.info("Nenhum Match (Reincidente) encontrado ainda. Deixe o Worker processar mais fotos.")
    else:
        st.success(f"Mostrando {matches_encontrados} grupos de indivíduos que apareceram em mais de uma foto/pasta.")

if __name__ == "__main__":
    carregar_painel()