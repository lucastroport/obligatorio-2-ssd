# 🚀 Guía Rápida de Setup - Retail 360 RAG Chatbot

Esta es una guía resumida para que tus compañeros puedan replicar el setup del proyecto rápidamente.

---

## ⚡ Setup Rápido (10-15 minutos)

### 📋 Pre-requisitos

- **Sistema Operativo**: Linux (Manjaro/Ubuntu) o macOS
- **Python**: 3.10 o superior
- **RAM**: Mínimo 8GB (recomendado 16GB)
- **Espacio en disco**: ~5GB libres

---

## 🔧 Pasos de Instalación

### 1. Clonar el Repositorio

```bash
git clone git@github.com:lucastroport/obligatorio-2-ssd.git
cd obligatorio-2-ssd
```

### 2. Instalar Ollama

**En Manjaro/Arch Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**En macOS:**
```bash
brew install ollama
```

**Verificar instalación:**
```bash
ollama --version
```

### 3. Descargar Modelos de IA

```bash
# Modelo principal para el chatbot
ollama pull llama3.2

# Modelo para embeddings
ollama pull nomic-embed-text

# Verificar que se descargaron
ollama list
```

### 4. Configurar Python

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
source venv/bin/activate  # En Linux/macOS
# o en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 5. Configurar Variables de Entorno

```bash
# Copiar archivo de configuración
cp .env.txt .env

# El archivo ya tiene las configuraciones correctas por defecto
# Si quieres cambiar algo, edita .env con tu editor favorito
```

### 6. Verificar que Todo Funciona

```bash
# Asegúrate de que el entorno virtual esté activado
source venv/bin/activate

# Ejecutar test del pipeline
python test_pipeline.py
```

**Salida esperada**: Deberías ver mensajes de log indicando que:
- ✅ Ollama está conectado
- ✅ Los datos se cargan correctamente
- ✅ El vector store se crea
- ✅ El chatbot responde a las preguntas de prueba

---

## 🎮 Cómo Usar el Chatbot

### Opción 1: API REST (Recomendado)

```bash
# Iniciar servidor
source venv/bin/activate
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Luego abre tu navegador en:
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

**Hacer una consulta desde la terminal:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuántos productos hay en el inventario?"}'
```

### Opción 2: Script Python

Crea un archivo `consultar.py`:

```python
from src.rag_pipeline import RAGPipeline

# Crear pipeline
pipeline = RAGPipeline(
    model_name="llama3.2",
    data_path="./data/TrabajoFinalPowerBI_v2.xlsx"
)

# Cargar vector store existente (rápido)
pipeline.initialize_from_existing_store()

# Hacer consultas
preguntas = [
    "¿Cuántos productos hay?",
    "¿Cuál es el cliente con más compras?",
    "¿Qué sucursal tuvo mejores ventas?"
]

for pregunta in preguntas:
    print(f"\n❓ {pregunta}")
    respuesta = pipeline.query_simple(pregunta)
    print(f"💬 {respuesta}\n")
```

Ejecutar:
```bash
python consultar.py
```

---

## 🔍 Estructura del Proyecto

```
obligatorio-2-ssd/
├── data/
│   └── TrabajoFinalPowerBI_v2.xlsx  # Datos del Power BI
├── src/
│   ├── app.py              # API REST con FastAPI
│   ├── data_loader.py      # Carga datos desde Excel
│   ├── embeddings.py       # Maneja embeddings y vector store
│   ├── rag_pipeline.py     # Pipeline RAG completo
│   └── utils/
│       └── logger.py       # Sistema de logging
├── test_pipeline.py        # Script de prueba
├── requirements.txt        # Dependencias Python
├── .env.txt               # Plantilla de configuración
└── README.md              # Documentación completa
```

---

## 🛠️ Troubleshooting

### Problema: "Ollama no está corriendo"

**Solución:**
```bash
# Verificar que Ollama está corriendo
ps aux | grep ollama

# Si no está corriendo, iniciarlo
ollama serve

# En otra terminal, probar
ollama run llama3.2 "Hola"
```

### Problema: "ModuleNotFoundError"

**Solución:**
```bash
# Asegurarse de estar en el entorno virtual
source venv/bin/activate

# Reinstalar dependencias
pip install -r requirements.txt
```

### Problema: "No se encuentra el archivo Excel"

**Solución:**
Verifica que el archivo esté en la ubicación correcta:
```bash
ls -la data/TrabajoFinalPowerBI_v2.xlsx
```

Si no está, asegúrate de tenerlo en la carpeta `data/`.

### Problema: Error de memoria con el modelo

**Solución:**
Si tienes poca RAM, usa un modelo más pequeño. Edita `.env`:
```bash
MODEL_NAME=llama3.2:1b  # Versión más pequeña
```

O descarga el modelo pequeño:
```bash
ollama pull llama3.2:1b
```

---

## 📊 Ejemplos de Preguntas que Puedes Hacer

- "¿Cuántos productos hay en total?"
- "¿Cuál fue el cliente que más compró?"
- "¿Qué producto es el más vendido?"
- "¿Cuántas ventas hubo en marzo de 2023?"
- "¿Qué sucursal tuvo mejores resultados?"
- "Muéstrame información sobre los clientes"

---

## 🎯 Próximos Pasos

1. **Probar diferentes preguntas** para entender qué datos tenemos
2. **Ajustar parámetros** en `.env` (temperatura, top_k, etc.)
3. **Agregar más datos** si es necesario
4. **Desarrollar frontend** (opcional, con React o Streamlit)

---

## 📚 Documentación Adicional

Para más detalles, consulta:
- `README.md` - Documentación completa
- `SETUP_ENVIRONMENT.md` - Guía técnica detallada
- `contexto-proyecto.md` - Contexto del obligatorio

---

## 💡 Tips

- **Primera vez**: La inicialización tarda ~30 segundos (crea el vector store)
- **Siguientes veces**: Usa `initialize_from_existing_store()` para cargar rápido
- **Cambios en datos**: Si modificas el Excel, borra `data/vectorstore/` y reinicializa
- **Mejor rendimiento**: Si tienes GPU NVIDIA, Ollama la usará automáticamente

---

## 🤝 Ayuda

Si tienes problemas:
1. Revisa la sección de Troubleshooting arriba
2. Consulta `SETUP_ENVIRONMENT.md` para la guía detallada
3. Verifica los logs en la terminal
4. Pregunta al equipo en el grupo

---

**¡Listo!** 🎉 Ya puedes empezar a trabajar con el chatbot.
