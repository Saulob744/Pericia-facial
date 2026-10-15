# ID-Hunter: Triagem Biométrica e Reconhecimento Facial Forense
### Polícia Científica do Estado do Paraná (SESP-PR) | Projeto PeríciaFace-PR

[![Status](https://img.shields.io/badge/Status-Produção%20Piloto%20(v2.4)-059669?style=for-the-badge&logo=shield)](DOCS/03_ARQUITETURA_E_COMPONENTES.md)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776ab?style=for-the-badge&logo=python)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Cluster-2496ed?style=for-the-badge&logo=docker)](docker-compose.yml)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL%2016-pgvector%20HNSW-336791?style=for-the-badge&logo=postgresql)](database_manager.py)
[![Celery](https://img.shields.io/badge/Celery-Distributed%20Workers-37814a?style=for-the-badge&logo=celery)](worker.py)

---

## 📌 Visão Geral do Sistema

O **ID-Hunter** é a plataforma pericial de inteligência visual desenvolvida para a **Polícia Científica do Paraná** e a **Secretaria de Estado da Segurança Pública (SESP-PR)**. O sistema atua como uma esteira automatizada para ingestão massiva, alinhamento facial, extração de assinaturas biométricas em espaço latente de 512 dimensões e cruzamento algorítmico de identidades criminais em mídias apreendidas (smartphones e cartões de memória).

Sob os preceitos de **Privacy by Design** e **Cadeia de Custódia Digital**, o sistema higieniza e analisa dezenas de milhares de fotos extraídas de dispositivos móveis por lote de apreensão, identificando com precisão matemática os **indivíduos reincidentes** que aparecem em múltiplos aparelhos celulares apreendidos em diferentes operações policiais.

---

## 🏛️ Arquitetura e Engenharia de Dados

A solução opera como um ecossistema de microsserviços dockerizados desacoplados:

```
[ Extrações de Celular ] ---> [ maestro_scanner.py ]
                                     |
                                     v (Lote atômico execute_values)
                           [ PostgreSQL 16 (fila_processamento) ]
                                     |
                                     v (Lock seguro FOR UPDATE SKIP LOCKED)
                              [ id_hunter.py (Streamlit UI) ]
                                     |
                                     v (Pacotes vetorizados de 32 imagens)
                           [ Redis 7 Broker (Fila Celery) ]
                                     |
                                     v (Consumo paralelo distribuído)
                         [ worker.py (Celery Autoscale 2 a 6) ]
                                     |
     +-------------------------------+-------------------------------+
     |                               |                               |
[ InsightFace Buffalo_L ]    [ YOLOv8n-Face Fallback ]   [ Resgate Moldura 35% ]
(ArcFace 512D direto)        (Rostos com oclusão)        (Ângulos extremos)
     +-------------------------------+-------------------------------+
                                     |
                                     v (Dupla deduplicação: IoU + Embedding)
                         [ database_manager.py ]
                                     |
                                     v (Normalização L2 + Busca Cosseno <=>)
                     [ PostgreSQL + pgvector (Índice HNSW) ]
                                     |
            +------------------------+------------------------+
            |                                                 |
            v                                                 v
   [ Tabela individuos ]                            [ Tabela ocorrencias ]
   (Vetor denso 512D único)                         (Hash SHA-256 + Lote + Crop)
```

---

## 📂 Documentação Oficial e Manuais Técnicos

Toda a documentação técnica, diagramas estruturados e manuais operacionais foram compilados na pasta [`DOCS/`](DOCS/):

| Documento | Formato | Descrição |
|---|---|---|
| [**DFD - Diagrama de Fluxo de Dados**](DOCS/01_DFD_DIAGRAMA_FLUXO_DADOS.md) | Markdown / Mermaid | Especificação completa do DFD Níveis 0, 1 e 2, Dicionário de Dados SQL, payloads Celery e mapeamento de data stores. |
| [**Manual do Usuário e Guia Operacional**](DOCS/02_MANUAL_DO_USUARIO_E_SISTEMA.md) | Markdown | Manual do Perito Criminal: passo a passo das 4 abas da UI, calibração de parâmetros, cálculo de ETA e troubleshooting. |
| [**Arquitetura Técnica e Componentes**](DOCS/03_ARQUITETURA_E_COMPONENTES.md) | Markdown | Especificação profunda arquivo por arquivo, modelos IA (ArcFace/YOLO), matriz de dependências e ciclo de vida dos dados. |
| [**Laudo e Documentação Unificada (PDF Executivo)**](DOCS/DOCUMENTACAO_SISTEMA_PERICIAFACE.pdf) | PDF (637 KB) | Documento formal consolidado com capa oficial, sumário executivo e todos os diagramas em alta resolução. |

### Diagramas Técnicos Gerados:
- [DFD Nível 0 - Diagrama de Contexto](DOCS/assets/dfd_nivel_0_contexto.png)
- [DFD Nível 1 - Decomposição em Processos](DOCS/assets/dfd_nivel_1_processos.png)
- [Fluxo Operacional do Perito (Jornada Ponta a Ponta)](DOCS/assets/fluxo_sistema_usuario.png)
- [Arquitetura de Microsserviços e Topologia Docker](DOCS/assets/arquitetura_pipeline.png)

---

## 🛠️ Especificação dos Arquivos do Projeto

```text
├── DOCS/                                # Documentação Oficial e Artefatos Forenses
│   ├── assets/                          # Diagramas técnicos em alta resolução (PNG)
│   ├── 01_DFD_DIAGRAMA_FLUXO_DADOS.md   # Especificação formal do DFD Níveis 0, 1 e 2
│   ├── 02_MANUAL_DO_USUARIO_E_SISTEMA.md# Guia de uso operacional para peritos
│   ├── 03_ARQUITETURA_E_COMPONENTES.md  # Especificação técnica de componentes e IA
│   ├── gerar_diagramas.py               # Script gerador de diagramas gráficos via PIL
│   ├── gerar_pdf_documentacao.py        # Compilador do manual em PDF executivo (fpdf2)
│   └── DOCUMENTACAO_SISTEMA_PERICIAFACE.pdf # Manual oficial consolidado em PDF
│
├── worker.py                            # Motor de IA Celery (InsightFace + YOLOv8n + Moldura)
├── database_manager.py                  # Camada PostgreSQL 16 + pgvector (Busca Cosseno HNSW)
├── id_hunter.py                         # Interface Web Pericial Streamlit (Métricas em Tempo Real)
├── maestro_scanner.py                   # Ingestão e mapeamento recursivo de mídias por lote
├── docker-compose.yml                   # Orquestração dos serviços (db, broker, web, worker)
├── dockerfile                           # Imagem otimizada Python 3.10-slim com OpenCV e PyTorch
├── requirements.txt                     # Dependências de pacotes com versões fixadas
└── yolov8n-face.pt                      # Pesos neurais pré-treinados para detecção facial robusta
```

---

## 🚀 Como Executar o Sistema

### Pré-requisitos
- **Docker Desktop** ou **Docker Engine** com suporte a Docker Compose.
- **Git** instalado.
- Mínimo recomendado: 4 núcleos de CPU e 8 GB de memória RAM (16 GB para acervos > 50.000 imagens).

### 1. Inicialização via Docker Compose (Recomendado para Produção)

1. Clone o repositório e acesse a pasta raiz:
   ```bash
   git clone https://github.com/engenheiroadilsoncabaca/facial_cientifica.git
   cd facial_cientifica
   ```

2. Inicie todos os microsserviços em segundo plano:
   ```bash
   docker compose up -d --build
   ```

3. Acesse os serviços nos seguintes endpoints:
   - **Interface Pericial (Streamlit):** [http://localhost:8501](http://localhost:8501)
   - **Banco Vetorial (PostgreSQL):** `localhost:5454` (Usuário: `perito`, Banco: `idhunter_db`)
   - **Broker de Mensagens (Redis):** `localhost:6379`

4. Para acompanhar os logs do cluster de IA em tempo real:
   ```bash
   docker compose logs -f worker
   ```

---

### 2. Execução Nativa / Ambiente de Desenvolvimento

Caso deseje executar os componentes fora do contêiner Docker:

```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Iniciar o worker Celery (em um terminal)
celery -A worker.app worker --loglevel=info --concurrency=4

# Iniciar a interface Web Streamlit (em outro terminal)
streamlit run id_hunter.py --server.port 8501
```

---

## 📊 Módulos da Interface Pericial

1. **Processamento de Fila:** Ingestão de mídias brutas, disparo de lotes vetorizados em pacotes de 32 imagens, monitoramento reativo a cada 3 segundos, cálculo dinâmico de velocidade (fotos/s e fotos/min) e previsão precisa de término (ETA).
2. **Auditoria de Reincidência:** Mapeamento automático de suspeitos que cruzam dois ou mais celulares apreendidos, com tags de apreensão e galeria de 12 miniaturas por indivíduo.
3. **Busca Biométrica Forense:** Upload de foto de referência de investigado com extração instantânea do vetor 512D e busca por similaridade cosseno no índice HNSW do PostgreSQL, retornando a acurácia percentual de certeza pericial.
4. **Curadoria de Descartes:** Auditoria visual de fotos descartadas por tamanho insuficiente (<60px) ou ausência de rosto, garantindo a integridade do processo investigativo.

---

## ⚖️ Licença e Uso Institucional

Projeto desenvolvido para a **Polícia Científica do Estado do Paraná (PCP-PR)** e **Secretaria de Estado da Segurança Pública do Paraná (SESP-PR)**. Uso reservado para atividades periciais, de inteligência criminal e persecução penal oficial.
