# 🏪 Retail 360 - Chatbot RAG con Ollama

Sistema de chatbot inteligente basado en RAG (Retrieval-Augmented Generation) para consultar datos de negocio de Retail 360 usando modelos locales con Ollama.

## 📋 Descripción

Este proyecto implementa un asistente de IA que permite realizar consultas en lenguaje natural sobre datos de ventas, productos, clientes y sucursales de la cadena Retail 360. Utiliza:

- **Ollama** para modelos de lenguaje locales
- **LangChain** para el pipeline RAG
- **Chroma/FAISS** para almacenamiento vectorial
- **FastAPI** para la API REST
- **Python 3.10+** como base del proyecto

## 🎯 Características

- ✅ Consultas en lenguaje natural sobre datos de negocio
- ✅ Respuestas basadas únicamente en los datos proporcionados (sin alucinaciones)
- ✅ Modelos de lenguaje ejecutados localmente
- ✅ API REST para integración con frontends
- ✅ Soporte para múltiples modelos (llama3, mistral, phi3, etc.)
- ✅ Vector store persistente (Chroma o FAISS)

## 📁 Estructura del Proyecto

```
obligatorio-2-ssd/
├── data/
│   ├── retail_360_dataset.xlsx      # Datos de negocio (Excel)
│   └── vectorstore/                 # Vector store persistente
│       ├── chroma/                  # Base de datos Chroma
│       └── faiss/                   # Índice FAISS
├── src/
│   ├── data_loader.py              # Cargador de datos desde Excel
│   ├── embeddings.py               # Gestión de embeddings y vector store
│   ├── rag_pipeline.py             # Pipeline RAG completo
│   ├── app.py                      # API REST con FastAPI
│   └── utils/
│       ├── __init__.py
│       └── logger.py               # Logging configurado
├── test_pipeline.py                # Script de prueba del pipeline
├── requirements.txt                # Dependencias Python
├── .env.txt                       # Plantilla de configuración
├── contexto-proyecto.md           # Contexto del obligatorio
├── setup-tecnico.md              # Guía de setup técnico
└── README.md                     # Este archivo
```

## 🚀 Instalación y Setup

### 1. Requisitos Previos

- Python 3.10 o superior
- Ollama instalado y corriendo
- 8GB+ de RAM recomendado

### 2. Instalar Ollama

**En Manjaro Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # Iniciar el daemon
```

**En macOS:**
```bash
brew install ollama
ollama serve
```

### 3. Descargar Modelos

```bash
# Modelo principal (recomendado)
ollama pull llama3

# Modelo para embeddings
ollama pull nomic-embed-text

# Alternativas
ollama pull mistral
ollama pull phi3
```

### 4. Configurar Entorno Python

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 5. Configurar Variables de Entorno

```bash
# Copiar plantilla
cp .env.txt .env

# Editar .env con tus configuraciones
nano .env
```

### 6. Preparar Datos

Coloca tu archivo Excel con los datos en:
```
data/retail_360_dataset.xlsx
```

## 🧪 Verificar Instalación

### Probar Ollama

```bash
# Verificar que Ollama está corriendo
ps aux | grep ollama

# Listar modelos disponibles
ollama list

# Probar un modelo
ollama run llama3 "¿Cuál es la capital de Uruguay?"
```

### Probar Pipeline RAG

```bash
python test_pipeline.py
```

Este script:
1. Verifica la conexión con Ollama
2. Carga los datos y crea el vector store
3. Ejecuta consultas de ejemplo

## 🎮 Uso

### Opción 1: API REST

```bash
# Iniciar servidor
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# O usando el script directamente
python src/app.py
```

La API estará disponible en `http://localhost:8000`

**Documentación interactiva:** `http://localhost:8000/docs`

#### Endpoints Principales

```bash
# Health check
curl http://localhost:8000/health

# Estado del sistema
curl http://localhost:8000/status

# Inicializar pipeline (primera vez)
curl -X POST http://localhost:8000/initialize

# Realizar consulta
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuántas ventas hubo en marzo de 2023?"}'

# Ejemplos de preguntas
curl http://localhost:8000/examples
```

### Opción 2: Script Python

```python
from src.rag_pipeline import create_rag_pipeline

# Crear e inicializar pipeline
pipeline = create_rag_pipeline(
    data_path="./data/retail_360_dataset.xlsx",
    model_name="llama3"
)

# Realizar consulta
response = pipeline.query_simple("¿Cuál fue el producto más vendido?")
print(response)
```

### Opción 3: Uso Interactivo

```python
from src.rag_pipeline import RAGPipeline

# Crear pipeline
pipeline = RAGPipeline(
    model_name="llama3",
    data_path="./data/retail_360_dataset.xlsx"
)

# Inicializar desde datos
pipeline.initialize_from_data()

# Realizar consultas
while True:
    pregunta = input("\nPregunta: ")
    if pregunta.lower() in ['salir', 'exit', 'quit']:
        break
    
    respuesta = pipeline.query_simple(pregunta)
    print(f"\nRespuesta: {respuesta}\n")
```

## 📊 Ejemplos de Consultas

```python
# Ventas
"¿Cuántas ventas hubo en marzo de 2023?"
"¿Cuál fue el total de ventas en 2023?"
"¿Qué mes tuvo más ventas?"

# Productos
"¿Cuál fue el producto más vendido?"
"¿Qué productos tienen mejor margen?"
"¿Cuántos productos diferentes se vendieron?"

# Clientes
"¿Cuál fue el cliente que más compró?"
"¿Cuántos clientes activos hay?"
"¿Quién es el cliente con mayor ticket promedio?"

# Sucursales
"¿Qué sucursal tuvo mejores resultados?"
"¿Cuál fue el crecimiento de la sucursal de Pocitos?"
"¿Qué local tiene más ventas?"
```

## ⚙️ Configuración Avanzada

### Cambiar Modelo

```python
pipeline = RAGPipeline(
    model_name="mistral",  # o "phi3", "llama3:70b", etc.
    data_path="./data/retail_360_dataset.xlsx"
)
```

### Ajustar Temperature

```python
# Más determinista (recomendado para datos)
pipeline.update_temperature(0.0)

# Más creativo
pipeline.update_temperature(0.7)
```

### Cambiar Vector Store

En `.env`:
```bash
VECTOR_STORE_TYPE=faiss  # o "chroma"
```

### Ajustar Número de Documentos Recuperados

```python
pipeline = RAGPipeline(
    top_k=10,  # Recuperar más documentos
    data_path="./data/retail_360_dataset.xlsx"
)
```

## 🔧 Troubleshooting

### Ollama no responde

```bash
# Verificar que está corriendo
systemctl status ollama  # Linux con systemd

# Reiniciar
ollama serve

# Verificar puerto
curl http://localhost:11434/api/version
```

### Error de memoria

```bash
# Usar un modelo más pequeño
ollama pull phi3:mini

# O ajustar el contexto en el código
```

### Vector store corrupto

```bash
# Eliminar y recrear
rm -rf data/vectorstore/*

# Ejecutar con force_recreate
python test_pipeline.py
```

## 📝 Desarrollo

### Estructura del Código

- **data_loader.py**: Carga datos desde Excel y los convierte en documentos
- **embeddings.py**: Genera embeddings y gestiona el vector store
- **rag_pipeline.py**: Pipeline completo de RAG (retrieval + generation)
- **app.py**: API REST con FastAPI

### Agregar Nuevos Datos

```python
from src.data_loader import DataLoader

loader = DataLoader("./data/nuevos_datos.xlsx")
loader.load_excel()
documents = loader.create_documents()

# Agregar al vector store existente
pipeline.embedding_manager.add_documents(documents)
```

## 📦 Dependencias Principales

- langchain >= 0.1.0
- langchain-community >= 0.0.10
- chromadb >= 0.4.22
- faiss-cpu >= 1.7.4
- fastapi >= 0.109.0
- pandas >= 2.1.4
- ollama >= 0.1.6

Ver `requirements.txt` para la lista completa.

## 🤝 Contribuir

Este es un proyecto académico para el curso de Sistema de Soporte de Decisión.

## 📄 Licencia

Proyecto académico - Universidad [Tu Universidad]

## 👥 Autores

- [Tu Nombre] - Desarrollo e implementación

## 🎓 Contexto Académico

Obligatorio de Inteligencia Artificial - Creación de Chatbot tipo RAG
Curso: Sistema de Soporte de Decisión
Año: 2025

---

**Nota**: Este proyecto utiliza modelos de lenguaje locales para garantizar privacidad y control de datos.
