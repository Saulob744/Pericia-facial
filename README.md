# Triagem Biométrica e Reconhecimento Facial (SESP-PR & CIENTIFICA) 

Este repositório contém o código-fonte da Prova de Conceito (PoC) do **ID-Hunter** (vinculado ao projeto PeríciaFace-PR). O sistema atua como um microsserviço de auditoria e triagem algorítmica projetado para a **Polícia Científica do Paraná** e para a **Secretaria de Segurança Pública (SESP-PR)**.

O objetivo principal é atuar como uma barreira preventiva estruturada sob o princípio de *Privacy by Design*, higienizando capturas faciais na origem (dispositivos de borda / *Edge Computing*) antes da submissão ao banco de dados estadual.

---

## O Problema e a Solução
Bancos de dados governamentais sofrem com a poluição visual gerada por cadastros duplicados e imagens com oclusões severas (máscaras, barbas espessas), o que reduz a eficácia da identificação criminal e eleva o risco de falsos positivos.

Para solucionar isso, desenvolvemos um *pipeline* operando em 4 camadas sequenciais estritamente via CPU:
1. **Higienização e Deduplicação:** Uso de SHA-256 e pHash para barrar redundâncias estruturais e imagens bit a bit idênticas.
2. **Triagem Semântica (YOLOv8-n):** Rede neural customizada para classificar as faces nas categorias *Limpo*, *Barba* e *Máscara*. Imagens ocluídas são descartadas imediatamente.
3. **Extração Geométrica (ArcFace):** Alinhamento facial via Transformação Afim (Warp Affine) e extração de vetores densos (*embeddings*) de 512 dimensões baseados em 5 marcos fiduciais.
4. **Clusterização Espacial (DBSCAN):** Agrupamento não supervisionado via distância euclidiana para auditar o espaço latente, isolando *outliers* e garantindo identidades únicas.

---

## Resultados Empíricos (Validação)
A arquitetura modular provou ser altamente eficiente para infraestruturas limitadas (*Edge Computing*):
* **Acurácia:** 100% de precisão preditiva no conjunto de validação de oclusões complexas (zero falsos positivos/negativos).
* **Eficiência de Hardware (CPU):** O YOLOv8-n customizado entregou latência de apenas **86,36 ms** por quadro, alocando ínfimos **3,10 MB de memória RAM**.
* **Governança:** Em um teste com 1.875 assinaturas vetorizadas, o algoritmo DBSCAN mapeou com sucesso 1.843 identidades únicas e isolou 32 anomalias (*outliers*).

---

##  Tecnologias Utilizadas
* **Linguagem:** Python 3.11
* **Visão Computacional e IA:** Ultralytics (YOLOv8), InsightFace (ArcFace), OpenCV.
* **Ciência de Dados (Clusterização):** Scikit-learn (DBSCAN), Matplotlib, Seaborn.
* **Interface Analítica:** Streamlit (para renderização de KPIs e Funil de Triagem).
* **Arquitetura:** C4 Model.

---

##  Como Executar Localmente (Versão Edge)

### Clonar o Repositório
```bash
git clone [https://github.com/engenheiroadilsoncabaca/facial_cientifica.git](https://github.com/engenheiroadilsoncabaca/facial_cientifica.git)
cd facial_cientifica

---

### CONFIGURAÇÃO DO AMBIENTE VIRTUAL

```bash
conda create -n clean python=3.11
conda activate clean
pip install -r requirements.txt

### COMO EXECUTAR

1. **Central de Comando (CLI)**:
   ```bash
   python id_hunter.py
   ```

2. **Painel Web (Streamlit)**:
   ```bash
   streamlit run painel_web.py
   ```

---

## 📁 Arquivos Essenciais da Aplicação

- `id_hunter.py` - Central de Comando interativa (CLI).
- `painel_web.py` - Painel Gráfico Web (Streamlit) de Auditoria e Busca por Referência.
- `buscar_referencia.py` - Motor de busca pericial por foto de referência e rede de lotes.
- `database_manager.py` - Gerenciador do Banco PostgreSQL + pgvector.
- `maestro_scanner.py` - Varredor de pastas de entrada e enfileirador de tarefas.
- `worker_operario.py` - Processador de IA e extrator biométrico ArcFace.

