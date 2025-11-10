

# ⚙️ SETUP TÉCNICO DEL PROYECTO RAG (Retail 360)

Este documento complementa el **contexto principal del obligatorio** y se enfoca exclusivamente en preparar el entorno técnico del chatbot basado en **Ollama + LangChain + Chroma/FAISS**.

---

## 🧩 OBJETIVO PARA CLAUDE

Necesito que generes **una guía completa de setup y verificación**, lista para ejecutar tanto en **macOS** como en **Manjaro Linux**, que deje el entorno funcional para comenzar a codificar el pipeline RAG.

---

## 🧰 PASOS QUE DEBE INCLUIR LA GUÍA

### 1. 🔹 Instalación de Ollama
- Comandos exactos para instalar Ollama en **macOS** y **Manjaro Linux**.
- Verificación de instalación (`ollama --version`).
- Descarga y gestión de modelos (`ollama pull llama3`, `ollama pull mistral`, etc.).
- Cómo iniciar y mantener el daemon de Ollama activo (`ollama serve`).

### 2. 🔹 Configuración del entorno Python
- Creación y activación de entorno virtual (`python -m venv venv` o `conda create -n retail360 python=3.11`).
- Instalación de dependencias desde `requirements.txt`.
- Recomendación sobre versiones compatibles (Python ≥ 3.10).

### 3. 🔹 Dependencias principales
Incluir en el archivo `requirements.txt`:
```

langchain
langchain-community
chromadb
faiss-cpu
pandas
python-dotenv
fastapi
uvicorn
openpyxl
tqdm

```

(Opcional: agregar `streamlit` o `gradio` si se elige interfaz web.)

### 4. 🔹 Configuración del archivo `.env`
Ejemplo de `.env.txt`:
```

VECTOR_STORE_PATH=./data/vectorstore
MODEL_NAME=llama3
DATA_PATH=./data/retail_360_dataset.xlsx
PORT=8000

````
Instrucciones para copiarlo como `.env` real:
```bash
cp .env.txt .env
````

### 5. 🔹 Verificación de Ollama + LangChain

Generar un test mínimo que Claude incluya en la guía:

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
response = llm.invoke("¿Cuál es la capital de Uruguay?")
print(response)
```

✅ Si responde correctamente (Montevideo), Ollama y LangChain están conectados.

### 6. 🔹 Validación del entorno

Incluir pasos para:

* Confirmar que Ollama está corriendo (`ps aux | grep ollama`).
* Probar carga de modelo (`ollama run mistral`).
* Crear un embedding de prueba:

  ```python
  from langchain_community.embeddings import OllamaEmbeddings

  embeddings = OllamaEmbeddings(model="mistral")
  vector = embeddings.embed_query("Ventas del año 2023")
  print(vector[:5])
  ```

### 7. 🔹 Recomendaciones finales

* Definir directorio de trabajo (`retail360-chatbot/`).
* Usar `uvicorn src.app:app --reload` si se implementa backend con FastAPI.
* Comandos de verificación rápida:

  ```bash
  ollama list
  pip list | grep langchain
  python --version
  ```

---

## 💡 RESULTADO FINAL ESPERADO

Al finalizar esta guía, el entorno debe permitir:

1. Ejecutar consultas locales al modelo Ollama con LangChain.
2. Crear y guardar embeddings en Chroma o FAISS.
3. Cargar datos del Excel original del Power BI.
4. Iniciar la app (FastAPI o interfaz) para probar el chatbot.

---

## 🧱 ESTRUCTURA FINAL RECOMENDADA DEL PROYECTO

```
retail360-chatbot/
├── data/
│   └── retail_360_dataset.xlsx
├── src/
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── rag_pipeline.py
│   ├── app.py
│   └── utils/
│       └── __init__.py
├── .env.txt
├── requirements.txt
├── SETUP_ENVIRONMENT.md
├── README.md
```


