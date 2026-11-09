# pdf_ollama

Procesa PDFs con IA local: extrae texto, lo indexa en vectores (FAISS) con embeddings de Ollama y responde preguntas con un LLM de Ollama.

## Visibilidad y rama

- Público.
- Rama por defecto: `main`.

## Stack

- Python 3.
- `requirements.txt`: `langchain`, `langchain_core`, `langchain_community`, `langchain_ollama`, `pdfplumber`, `faiss-cpu`. Sin versiones fijadas.
- Backend: servidor Ollama (`http://toletum:11434`) con el modelo `deepseek-r1:8b`.
- `langchain_text_splitters` se usa en el código pero no está en `requirements.txt` (no verificado si llega como dependencia transitiva).

## Estructura

- `main.py`: ejemplo 1. Carga `pdf/JoseCarlosSanchezGomezES.pdf` en memoria (`InMemoryVectorStore`), hace una búsqueda y responde con el LLM.
- `pdf2vectors.py`: ejemplo 2. Indexa los PDFs de `pdf/` en FAISS guardado en `./data`, evitando reprocesar los ya indexados.
- `fromlocal.py`: carga el índice de `./data` y responde preguntas con el LLM.
- `checkpdf.py`: comprueba el índice FAISS en `./data` (no revisado en detalle).
- `README.md`: instrucciones de uso.
- `.gitignore`: `venv/`, `pdf/*`, `data/*`.

## Comandos

Del README y del código (no ejecutados):

- Entorno: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Preparar: crear `pdf/` y `data/`.
- Ejemplo 1: `python main.py`
- Ejemplo 2: `python pdf2vectors.py` y después `python fromlocal.py`.
- Comprobar índice: `python checkpdf.py`.
- Tests: no hay.

## Configuración

- Sin variables de entorno. La URL de Ollama y el modelo están codificados en cada script.
- Rutas codificadas: `pdf/`, `./data`, y un nombre de PDF concreto en `main.py`.
- Requisito: servidor Ollama con `deepseek-r1:8b` y modelo de embeddings disponible.

## Convenciones para agentes

- Mover `http://toletum:11434` y `deepseek-r1:8b` a variables o constantes compartidas si se cambian en varios scripts.
- `FAISS.load_local(..., allow_dangerous_deserialization=True)`: solo cargar índices generados por el propio repo.
- Los PDFs y el índice `data/` no se versionan (`.gitignore`).
- Las respuestas del modelo `deepseek-r1` incluyen bloques `<think>`: no asumir respuesta limpia.

## Estado

- Sin tests.
- Atención: `main.py` tiene una asignación duplicada (`text_splitter = text_splitter = ...`).
- Atención: `main.py` y `fromlocal.py` tienen lógica casi idéntica; el código es de prueba.
- Atención: `main.py` usa un PDF personal concreto (nombre de persona); no commitear PDFs nuevos.
- Atención: `allow_dangerous_deserialization=True` (riesgo si el índice es externo).
- Las dependencias no tienen versiones fijadas.
