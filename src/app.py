"""
API REST con FastAPI para el chatbot Retail 360.
Expone endpoints para consultas al RAG y gestión del sistema.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline
from src.utils.logger import logger

# Cargar variables de entorno
load_dotenv()

# Configuración
DATA_PATH = os.getenv("DATA_PATH", "./data/retail_360_dataset.xlsx")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vectorstore")
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Crear aplicación FastAPI
app = FastAPI(
    title="Retail 360 RAG API",
    description="API del chatbot inteligente para consultas de datos de negocio de Retail 360",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline RAG global
rag_pipeline: Optional[RAGPipeline] = None


# ===== MODELOS PYDANTIC =====

class QueryRequest(BaseModel):
    """Modelo para solicitud de consulta"""
    question: str = Field(..., description="Pregunta en lenguaje natural", min_length=3)
    return_sources: bool = Field(False, description="Si incluir documentos fuente")
    k: Optional[int] = Field(None, description="Número de documentos a recuperar")


class QueryResponse(BaseModel):
    """Modelo para respuesta de consulta"""
    question: str
    answer: str
    timestamp: str
    source_documents: Optional[List[Dict[str, Any]]] = None


class InitializeRequest(BaseModel):
    """Modelo para inicialización del pipeline"""
    force_recreate: bool = Field(False, description="Forzar recreación del vector store")
    sheet_names: Optional[List[str]] = Field(None, description="Hojas específicas a cargar")


class StatusResponse(BaseModel):
    """Modelo para estado del sistema"""
    status: str
    is_initialized: bool
    model_name: str
    temperature: float
    top_k: int
    vector_store_type: str


class HealthResponse(BaseModel):
    """Modelo para health check"""
    status: str
    message: str
    timestamp: str


# ===== EVENTOS DE INICIO/CIERRE =====

@app.on_event("startup")
async def startup_event():
    """Inicializa el pipeline al arrancar la aplicación"""
    global rag_pipeline
    
    logger.info("=== Iniciando Retail 360 RAG API ===")
    
    try:
        # Crear pipeline
        rag_pipeline = RAGPipeline(
            model_name=MODEL_NAME,
            embedding_model=EMBEDDING_MODEL,
            data_path=DATA_PATH,
            vector_store_path=VECTOR_STORE_PATH,
            vector_store_type=VECTOR_STORE_TYPE,
            temperature=TEMPERATURE,
            top_k=TOP_K
        )
        
        # Intentar cargar vector store existente
        vector_store_path = Path(VECTOR_STORE_PATH)
        if VECTOR_STORE_TYPE == "chroma":
            store_exists = (vector_store_path / "chroma").exists()
        else:  # faiss
            store_exists = (vector_store_path / "faiss" / "index.faiss").exists()
        
        if store_exists:
            logger.info("Vector store existente encontrado, cargando...")
            rag_pipeline.initialize_from_existing_store()
        else:
            logger.info("No se encontró vector store, será necesario inicializar con datos")
        
        logger.info("✓ API iniciada exitosamente")
        
    except Exception as e:
        logger.error(f"Error al iniciar la API: {e}")
        logger.warning("API iniciada pero el pipeline RAG necesita ser inicializado manualmente")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la aplicación"""
    logger.info("=== Cerrando Retail 360 RAG API ===")


# ===== ENDPOINTS =====

@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint raíz - Health check"""
    return HealthResponse(
        status="ok",
        message="Retail 360 RAG API está funcionando",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio"""
    return HealthResponse(
        status="healthy",
        message="Servicio operativo",
        timestamp=datetime.now().isoformat()
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Obtiene el estado del sistema"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")
    
    stats = rag_pipeline.get_stats()
    
    return StatusResponse(
        status="initialized" if stats["is_initialized"] else "not_initialized",
        is_initialized=stats["is_initialized"],
        model_name=stats["model_name"],
        temperature=stats["temperature"],
        top_k=stats["top_k"],
        vector_store_type=stats["vector_store_type"]
    )


@app.post("/initialize")
async def initialize_pipeline(request: InitializeRequest, background_tasks: BackgroundTasks):
    """
    Inicializa el pipeline desde los datos del Excel.
    Puede tardar varios minutos si force_recreate=True.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline no disponible")
    
    if rag_pipeline.is_initialized and not request.force_recreate:
        return {
            "message": "Pipeline ya está inicializado",
            "status": "ok"
        }
    
    try:
        logger.info("Iniciando inicialización del pipeline...")
        
        # Ejecutar inicialización
        rag_pipeline.initialize_from_data(
            sheet_names=request.sheet_names,
            force_recreate=request.force_recreate
        )
        
        return {
            "message": "Pipeline inicializado exitosamente",
            "status": "ok",
            "force_recreate": request.force_recreate
        }
        
    except Exception as e:
        logger.error(f"Error al inicializar pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error al inicializar: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Realiza una consulta al chatbot RAG.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline no disponible")
    
    if not rag_pipeline.is_initialized:
        raise HTTPException(
            status_code=400,
            detail="Pipeline no inicializado. Use POST /initialize primero"
        )
    
    try:
        # Realizar consulta
        result = rag_pipeline.query(
            question=request.question,
            return_sources=request.return_sources
        )
        
        # Formatear respuesta
        response = QueryResponse(
            question=result["question"],
            answer=result["answer"],
            timestamp=datetime.now().isoformat()
        )
        
        # Agregar fuentes si se solicitaron
        if request.return_sources and "source_documents" in result:
            response.source_documents = [
                {
                    "content": doc.page_content[:500],  # Limitar tamaño
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error al procesar consulta: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")


@app.get("/documents/search")
async def search_documents(query: str, k: int = 5):
    """
    Busca documentos relevantes sin generar respuesta.
    """
    if not rag_pipeline or not rag_pipeline.is_initialized:
        raise HTTPException(status_code=400, detail="Pipeline no inicializado")
    
    try:
        documents = rag_pipeline.get_relevant_documents(query, k=k)
        
        return {
            "query": query,
            "count": len(documents),
            "documents": [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        }
        
    except Exception as e:
        logger.error(f"Error al buscar documentos: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/test-connection")
async def test_connection():
    """
    Prueba la conexión con Ollama.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline no disponible")
    
    try:
        success = rag_pipeline.test_connection()
        
        if success:
            return {
                "status": "ok",
                "message": "Conexión con Ollama exitosa",
                "model": rag_pipeline.model_name
            }
        else:
            raise HTTPException(status_code=500, detail="Error en la conexión con Ollama")
            
    except Exception as e:
        logger.error(f"Error al probar conexión: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.put("/config/temperature")
async def update_temperature(temperature: float):
    """
    Actualiza la temperatura del modelo (0.0 - 1.0).
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline no disponible")
    
    if not 0.0 <= temperature <= 1.0:
        raise HTTPException(status_code=400, detail="Temperature debe estar entre 0.0 y 1.0")
    
    rag_pipeline.update_temperature(temperature)
    
    return {
        "message": "Temperature actualizada",
        "temperature": temperature
    }


# ===== EJEMPLOS DE CONSULTAS =====

@app.get("/examples")
async def get_examples():
    """
    Retorna ejemplos de preguntas que se pueden hacer.
    """
    examples = [
        "¿Cuántas ventas hubo en marzo de 2023?",
        "¿Cuál fue el cliente que más compró?",
        "¿Qué producto fue el más vendido en Pocitos?",
        "¿Qué sucursal tuvo el mayor crecimiento entre 2022 y 2023?",
        "¿Cuál es el promedio de ventas por mes?",
        "¿Qué productos tienen mejor margen de ganancia?",
        "¿Cuántos clientes activos hay?",
        "¿Cuál es la sucursal con más ventas?"
    ]
    
    return {
        "examples": examples,
        "note": "Estas son preguntas de ejemplo. Las respuestas dependen de los datos cargados."
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Iniciando servidor en {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
