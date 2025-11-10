# 🔧 GUÍA DE SETUP TÉCNICO - RETAIL 360 RAG

Guía completa para configurar el entorno técnico del chatbot basado en **Ollama + LangChain + Chroma/FAISS** en **macOS** y **Manjaro Linux**.

---

## 📋 TABLA DE CONTENIDOS

1. [Instalación de Ollama](#1-instalación-de-ollama)
2. [Configuración del entorno Python](#2-configuración-del-entorno-python)
3. [Instalación de dependencias](#3-instalación-de-dependencias)
4. [Configuración del archivo .env](#4-configuración-del-archivo-env)
5. [Verificación de Ollama + LangChain](#5-verificación-de-ollama--langchain)
6. [Validación del entorno](#6-validación-del-entorno)
7. [Comandos de verificación](#7-comandos-de-verificación)

---

## 1. 🔹 INSTALACIÓN DE OLLAMA

### En Manjaro Linux

```bash
# Método 1: Script oficial (recomendado)
curl -fsSL https://ollama.com/install.sh | sh

# Método 2: Desde AUR
yay -S ollama

# Verificar instalación
ollama --version

# Iniciar servicio
sudo systemctl start ollama
sudo systemctl enable ollama  # Para inicio automático

# O ejecutar manualmente
ollama serve
```

### En macOS

```bash
# Método 1: Homebrew (recomendado)
brew install ollama

# Método 2: Descarga directa
# Visitar: https://ollama.com/download

# Verificar instalación
ollama --version

# Iniciar Ollama
ollama serve
```

### Descargar Modelos

```bash
# Modelo principal (LLM para generación)
ollama pull llama3

# Modelo para embeddings
ollama pull nomic-embed-text

# Alternativas de LLM
ollama pull mistral
ollama pull phi3
ollama pull codellama

# Listar modelos instalados
ollama list

# Probar un modelo
ollama run llama3 "Hola, ¿cómo estás?"
```

### Mantener Ollama Activo

```bash
# Ver si Ollama está corriendo
ps aux | grep ollama

# En Manjaro (con systemd)
sudo systemctl status ollama
sudo systemctl start ollama
sudo systemctl restart ollama

# Ejecución manual (ambos sistemas)
ollama serve

# Verificar que responde
curl http://localhost:11434/api/version
```

**Nota**: Ollama debe estar corriendo en segundo plano antes de usar el chatbot.

---

## 2. 🔹 CONFIGURACIÓN DEL ENTORNO PYTHON

### Verificar Python

```bash
# Verificar versión (debe ser >= 3.10)
python --version
python3 --version

# Si no tienes Python 3.10+, instalar:

# En Manjaro
sudo pacman -S python python-pip

# En macOS
brew install python@3.11
```

### Crear Entorno Virtual

```bash
# Navegar al directorio del proyecto
cd /home/lucas/Desktop/obligatorio-2-ssd

# Método 1: venv (recomendado)
python -m venv venv

# Activar entorno
# En Linux/macOS
source venv/bin/activate

# En Windows (si aplica)
venv\Scripts\activate

# Método 2: conda (alternativa)
conda create -n retail360 python=3.11
conda activate retail360

# Verificar que estás en el entorno
which python  # Debe apuntar a tu venv
pip --version
```

**Importante**: Siempre activa el entorno antes de trabajar:
```bash
source venv/bin/activate  # o conda activate retail360
```

---

## 3. 🔹 INSTALACIÓN DE DEPENDENCIAS

```bash
# Asegurarse de estar en el entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar todas las dependencias
pip install -r requirements.txt

# Verificar instalaciones principales
pip list | grep langchain
pip list | grep chromadb
pip list | grep faiss
pip list | grep fastapi

# Si hay errores, instalar manualmente:
pip install langchain langchain-community
pip install chromadb
pip install faiss-cpu
pip install pandas openpyxl
pip install fastapi uvicorn
pip install python-dotenv
pip install ollama
```

### Troubleshooting: Problemas Comunes

```bash
# Error con faiss en macOS M1/M2
pip install faiss-cpu --no-cache-dir

# Error con ChromaDB
pip install chromadb --no-binary chromadb

# Dependencias de sistema (Manjaro)
sudo pacman -S gcc python-pip

# Dependencias de sistema (macOS)
xcode-select --install
```

---

## 4. 🔹 CONFIGURACIÓN DEL ARCHIVO .env

```bash
# Copiar plantilla
cp .env.txt .env

# Editar con tu editor favorito
nano .env
# o
vim .env
# o
code .env
```

### Configuración Recomendada

```bash
# ============================================
# CONFIGURACIÓN RECOMENDADA PARA INICIO
# ============================================

# Rutas del proyecto
VECTOR_STORE_PATH=./data/vectorstore
DATA_PATH=./data/retail_360_dataset.xlsx

# Modelos de Ollama
MODEL_NAME=llama3
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

# Vector store (elegir uno)
VECTOR_STORE_TYPE=chroma
# Alternativa: VECTOR_STORE_TYPE=faiss

# Configuración del RAG
TOP_K_RESULTS=5
TEMPERATURE=0.1
MAX_TOKENS=2000

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Logging
LOG_LEVEL=INFO
```

### Modelos Alternativos

```bash
# Para equipos con menos recursos
MODEL_NAME=phi3:mini
EMBEDDING_MODEL=nomic-embed-text

# Para mejor calidad (requiere más RAM)
MODEL_NAME=llama3:70b
EMBEDDING_MODEL=mxbai-embed-large

# Para respuestas más rápidas
MODEL_NAME=mistral
EMBEDDING_MODEL=nomic-embed-text
```

---

## 5. 🔹 VERIFICACIÓN DE OLLAMA + LANGCHAIN

### Test Básico de Ollama

```bash
# Verificar que Ollama responde
curl http://localhost:11434/api/version

# Probar generación de texto
ollama run llama3 "¿Cuál es la capital de Uruguay?"
```

### Test de LangChain con Ollama

Crear archivo `test_langchain.py`:

```python
from langchain_community.llms import Ollama

# Probar conexión
llm = Ollama(model="llama3")
response = llm.invoke("¿Cuál es la capital de Uruguay?")
print(response)
```

Ejecutar:
```bash
python test_langchain.py
```

**Resultado esperado**: Debe responder "Montevideo" o similar.

---

## 6. 🔹 VALIDACIÓN DEL ENTORNO

### Test de Embeddings

Crear archivo `test_embeddings.py`:

```python
from langchain_community.embeddings import OllamaEmbeddings

# Crear embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Probar
text = "Ventas del año 2023"
vector = embeddings.embed_query(text)

print(f"Dimensión del vector: {len(vector)}")
print(f"Primeros 5 valores: {vector[:5]}")
```

Ejecutar:
```bash
python test_embeddings.py
```

**Resultado esperado**: Debe mostrar un vector de ~768 dimensiones.

### Test del Pipeline Completo

```bash
# Ejecutar script de prueba completo
python test_pipeline.py
```

Este script:
1. ✅ Verifica conexión con Ollama
2. ✅ Carga datos desde Excel
3. ✅ Crea el vector store
4. ✅ Ejecuta consultas de ejemplo

---

## 7. 🔹 COMANDOS DE VERIFICACIÓN

### Verificación Rápida del Sistema

```bash
# 1. Verificar Python
python --version

# 2. Verificar que estás en el entorno virtual
which python  # Debe apuntar a venv/bin/python

# 3. Verificar Ollama
ps aux | grep ollama
ollama list

# 4. Verificar dependencias
pip list | grep -E "langchain|chromadb|faiss|fastapi"

# 5. Verificar estructura de archivos
ls -la data/
ls -la src/

# 6. Probar Ollama
curl http://localhost:11434/api/version

# 7. Ejecutar test básico
python -c "from langchain_community.llms import Ollama; print('OK')"
```

### Script de Verificación Completo

Crear `verify_setup.sh`:

```bash
#!/bin/bash

echo "==================================="
echo "VERIFICACIÓN DE SETUP - RETAIL 360"
echo "==================================="

echo -e "\n1. Verificando Python..."
python --version

echo -e "\n2. Verificando entorno virtual..."
which python

echo -e "\n3. Verificando Ollama..."
if ps aux | grep -q "[o]llama"; then
    echo "✓ Ollama está corriendo"
else
    echo "✗ Ollama NO está corriendo"
    echo "  Ejecutar: ollama serve"
fi

echo -e "\n4. Verificando modelos de Ollama..."
ollama list

echo -e "\n5. Verificando dependencias Python..."
pip list | grep -E "langchain|chromadb|faiss|fastapi|pandas"

echo -e "\n6. Verificando estructura de archivos..."
if [ -f "data/retail_360_dataset.xlsx" ]; then
    echo "✓ Archivo de datos encontrado"
else
    echo "✗ Archivo de datos NO encontrado"
    echo "  Colocar Excel en: data/retail_360_dataset.xlsx"
fi

echo -e "\n7. Verificando conexión Ollama..."
curl -s http://localhost:11434/api/version | python -m json.tool

echo -e "\n==================================="
echo "VERIFICACIÓN COMPLETADA"
echo "==================================="
```

Ejecutar:
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

---

## 💡 RESULTADO FINAL ESPERADO

Después de completar esta guía, deberías poder:

✅ **Ejecutar Ollama localmente**
```bash
ollama serve
ollama run llama3 "test"
```

✅ **Crear embeddings**
```bash
python test_embeddings.py
```

✅ **Cargar datos del Excel**
```python
from src.data_loader import DataLoader
loader = DataLoader("./data/retail_360_dataset.xlsx")
loader.load_excel()
```

✅ **Iniciar el servidor API**
```bash
uvicorn src.app:app --reload
```

✅ **Realizar consultas al chatbot**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuántas ventas hubo?"}'
```

---

## 🆘 TROUBLESHOOTING COMÚN

### Problema: "Connection refused" con Ollama

**Solución:**
```bash
# Verificar puerto
netstat -tulpn | grep 11434

# Reiniciar Ollama
killall ollama
ollama serve
```

### Problema: "Model not found"

**Solución:**
```bash
# Descargar el modelo
ollama pull llama3
ollama pull nomic-embed-text

# Verificar
ollama list
```

### Problema: Error de memoria

**Solución:**
```bash
# Usar modelo más pequeño
ollama pull phi3:mini

# Modificar .env
MODEL_NAME=phi3:mini
```

### Problema: Import errors de LangChain

**Solución:**
```bash
# Reinstalar dependencias
pip uninstall langchain langchain-community
pip install langchain langchain-community --upgrade
```

---

## 📚 RECURSOS ADICIONALES

- **Ollama**: https://ollama.com/
- **LangChain**: https://python.langchain.com/
- **ChromaDB**: https://www.trychroma.com/
- **FastAPI**: https://fastapi.tiangolo.com/

---

## ✅ CHECKLIST FINAL

Antes de comenzar a codificar, verifica:

- [ ] Ollama instalado y corriendo
- [ ] Modelos descargados (llama3, nomic-embed-text)
- [ ] Entorno virtual Python activado
- [ ] Dependencias instaladas (requirements.txt)
- [ ] Archivo .env configurado
- [ ] Datos de Excel en data/
- [ ] Test de LangChain exitoso
- [ ] Test de embeddings exitoso
- [ ] Estructura de carpetas creada

**¡Listo para desarrollar! 🚀**
