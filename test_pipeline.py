"""
Script de prueba para verificar el pipeline RAG.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.utils.logger import logger
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

def main():
    """Función principal de prueba"""
    
    print("\n" + "="*60)
    print("PRUEBA DEL PIPELINE RAG - RETAIL 360")
    print("="*60 + "\n")
    
    # Configuración
    DATA_PATH = os.getenv("DATA_PATH", "./data/retail_360_dataset.xlsx")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
    
    # Crear pipeline
    logger.info("Creando pipeline RAG...")
    pipeline = RAGPipeline(
        model_name=MODEL_NAME,
        data_path=DATA_PATH
    )
    
    # Probar conexión
    logger.info("\n1. Probando conexión con Ollama...")
    if not pipeline.test_connection():
        logger.error("No se pudo conectar con Ollama. Asegúrate de que esté corriendo.")
        return
    
    # Inicializar pipeline
    logger.info("\n2. Inicializando pipeline (puede tardar unos minutos)...")
    try:
        pipeline.initialize_from_data(force_recreate=False)
    except Exception as e:
        logger.error(f"Error al inicializar: {e}")
        return
    
    # Consultas de ejemplo
    logger.info("\n3. Probando consultas...")
    
    queries = [
        "¿Cuántas ventas hubo en total?",
        "¿Cuál fue el producto más vendido?",
        "¿Qué sucursal tuvo mejores resultados?",
    ]
    
    for query in queries:
        print("\n" + "-"*60)
        print(f"Pregunta: {query}")
        print("-"*60)
        
        try:
            result = pipeline.query(query, return_sources=True)
            print(f"\nRespuesta:\n{result['answer']}\n")
            
            if result.get('source_documents'):
                print(f"Documentos consultados: {len(result['source_documents'])}")
                
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
    
    print("\n" + "="*60)
    print("PRUEBA COMPLETADA")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
