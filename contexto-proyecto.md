# 🎯 CONTEXTO DEL PROYECTO

Estamos desarrollando el **Obligatorio de Inteligencia Artificial - Creación de Chatbot tipo RAG** para el curso de *Sistema de Soporte de Decisión*.  
El caso se basa en la empresa ficticia **Retail 360**, una cadena regional que quiere un asistente inteligente para consultar datos de negocio (ventas, productos, clientes, locales, etc.) en lenguaje natural.

El chatbot debe responder preguntas como:
- ¿Cuántas ventas hubo en marzo de 2023?
- ¿Cuál fue el cliente que más compró?
- ¿Qué producto fue el más vendido en Pocitos?
- ¿Qué sucursal tuvo el mayor crecimiento entre 2022 y 2023?

### ⚙️ REQUISITOS TÉCNICOS
El proyecto debe:
1. Procesar los datos del obligatorio anterior de Power BI (Excel con tablas de ventas, productos, clientes, locales, etc.).
2. Convertirlos en texto o documentos indexables.
3. Crear un **repositorio vectorial** (FAISS, Chroma o Milvus).
4. Conectarlo con un modelo de lenguaje mediante **LangChain** u otra librería.
5. Usar preferentemente un **modelo local con Ollama** (por ejemplo, `llama3`, `mistral`, `phi3`).
6. Desarrollar una aplicación (frontend o notebook) que permita escribir preguntas y obtener respuestas.
7. Asegurar que las respuestas provengan **solo del dataset** (sin inventar datos externos).

### 💻 ENTORNO DISPONIBLE
- **MacBook** (macOS)
- **PC de escritorio con Manjaro Linux**
- Ambos con posibilidad de instalar **Ollama**
- Familiaridad con Python, FastAPI/Flask, LangChain, React y herramientas locales.

### 📦 ENTREGA FINAL
El entregable es un **PDF con**:
- Carátula
- Enlace a repositorio GitHub
- Documentación de endpoints
- Resumen del trabajo técnico
- Manual de ejecución
- Enlace a video demostrativo (máx. 5 minutos)

---

# 🧩 OBJETIVO PARA CLAUDE

Necesito que me ayudes a construir paso a paso el proyecto **desde cero**, enfocado en la implementación técnica del chatbot RAG con **Ollama + LangChain + Chroma/FAISS**.

### 🎯 PRIMERA ETAPA (lo que necesito que generes ahora)
1. **Estructura completa del proyecto (carpetas y archivos)** con nombres sugeridos, por ejemplo:
```text
├── data/
│ └── TrabajoFinalPowerBI_v2.xlsx
├── src/
│ ├── data_loader.py
│ ├── embeddings.py
│ ├── rag_pipeline.py
│ ├── app.py
│ └── utils/
├── requirements.txt
├── .env.txt
├── README.md
```

2. **Explicación de flujo completo del RAG:**
- Cargar y transformar datos desde Excel.
- Crear documentos/chunks con metadatos.
- Generar embeddings y guardarlos en el vector store.
- Conectar el modelo Ollama vía LangChain.
- Pipeline de recuperación y generación.
3. Código base para cada uno de los módulos, preparado para ir completando.
4. Si es posible, incluir ejemplos de consultas tipo:
```python
query = "¿Cuál fue el producto más vendido en marzo de 2023?"
response = rag.query(query)
print(response)
```
