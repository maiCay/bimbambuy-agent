# BimBam Buy - Agente Virtual Inteligente

Agente conversacional RAG (Retrieval-Augmented Generation) que responde preguntas de clientes de **BimBam Buy** utilizando la documentación oficial como base de conocimiento.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                  Interfaz Gradio                        │
│         (chat + streaming + preguntas sugeridas)        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Grafo LangGraph (agent.py)                 │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐         │
│  │ classify │──▶│ retrieve │──▶│   generate   │         │
│  │(keywords)│   │  (FAISS) │   │  (Gemini)    │         │
│  └──────────┘   └──────────┘   └──────────────┘         │
│       │              │                │                 │
│       ▼              ▼                ▼                 │
│  Clasifica      Busca docs      Genera la               │
│  por keywords   en paralelo     respuesta (streaming)   │
│  (instantáneo)                                          │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Vector Store FAISS (knowledge_base.py)        │
│                                                         │
│  Embeddings: Gemini gemini-embedding-001                │
│  Fuentes:                                               │
│   ├── 5 PDFs (Docs/)                                    │
│   │   ├── Política de Reembolsos                        │
│   │   ├── Programa de Afiliados                         │
│   │   ├── Guía de Tiempos y Costos de Envío             │
│   │   ├── Preguntas Frecuentes sobre Pagos              │
│   │   └── Manual de Garantía                            │
│   └── CSV (data/catalogo_productos.csv)                 │
│       └── Catálogo de 15 productos                      │
└─────────────────────────────────────────────────────────┘
```

## Flujo de ejecución

![Grafo del agente](graph.png)

1. El cliente escribe una pregunta en la interfaz Gradio
2. El nodo **classify** clasifica la pregunta por palabras clave en: envíos, pagos, garantía, productos, afiliados o general (sin llamada a LLM)
3. El nodo **retrieve** recupera los chunks más relevantes del vector store FAISS de forma paralela
4. El nodo **generate** (Gemini) genera una respuesta streaming usando el contexto recuperado
5. La respuesta se muestra token por token al cliente

## Tecnologías

| Componente | Tecnología |
|------------|------------|
| Orquestación | LangGraph |
| LLM | Google Gemini 3.5 Flash Lite |
| Embeddings | Gemini gemini-embedding-001 |
| Vector Store | FAISS (local) |
| Interfaz | Gradio |
| Framework | LangChain |

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/bimbambuy-agent.git
cd bimbambuy-agent

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
# Editar .env y colocar tu GEMINI_API_KEY
```

## Uso

```bash
# Ejecutar la aplicación
python app.py
```

Abrir http://localhost:7860 en el navegador.

### Ver el grafo del agente

```bash
python agent.py
```

Genera un archivo `graph.png` con la visualización del grafo LangGraph.

### Ejecutar tests de ejemplo

```bash
python -c "from agent import run_tests; run_tests()"
```

### Primera ejecución

En la primera ejecución, el sistema:
1. Lee los 5 PDFs de la carpeta `Docs/`
2. Lee el CSV de la carpeta `data/`
3. Genera los chunks de texto (500 caracteres)
4. Calcula los embeddings con Gemini
5. Crea el índice FAISS en la carpeta `faiss_index/`

Las siguientes ejecuciones cargan el índice pre-computado.

## Estructura del proyecto

```
├── Docs/                          # Documentación PDF de BimBam Buy
│   ├── Política de Reembolsos.pdf
│   ├── Programa de Afiliados.pdf
│   ├── Guía de Tiempos y Costos de Envío.pdf
│   ├── Preguntas Frecuentes.pdf
│   └── Manual de Garantía.pdf
├── data/
│   └── catalogo_productos.csv     # Catálogo de productos (15 items)
├── faiss_index/                   # Índice FAISS (generado automáticamente)
├── knowledge_base.py              # Carga y procesamiento de documentos → Vector Store
├── agent.py                       # Grafo LangGraph (classify → retrieve → generate)
├── app.py                         # Interfaz Gradio con streaming
├── graph.png                      # Visualización del grafo (generado con python agent.py)
├── requirements.txt               # Dependencias
├── .env                           # Variables de entorno (GEMINI_API_KEY)
└── README.md                      # Este archivo
```

## Conceptos aplicados

- **LangGraph**: Orquestación del agente con grafo de estados (StateGraph)
- **RAG (Retrieval-Augmented Generation)**: Recuperación de contexto relevante antes de generar respuestas
- **Embeddings**: Representación vectorial del contenido para búsqueda semántica
- **Vector Store (FAISS)**: Almacenamiento y búsqueda de similitud de documentos
- **Clasificación por keywords**: Enrutamiento rápido basado en reglas (sin LLM)
- **Streaming**: Respuestas token por token para mejor percepción de velocidad
- **Caché**: Almacenamiento de respuestas para preguntas repetidas
- **Concurrencia**: Búsquedas FAISS paralelas con ThreadPoolExecutor
- **Prompt Engineering**: Prompts especializados para generación de respuestas

## Variables de entorno

| Variable | Descripción | Obtener en |
|---|---|---|
| `GEMINI_API_KEY` | API key de Google Gemini | [Google AI Studio](https://aistudio.google.com/apikey) |
