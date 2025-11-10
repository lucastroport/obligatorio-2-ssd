"""
Pipeline RAG completo para Retail 360.
Integra el cargador de datos, embeddings y el modelo LLM de Ollama.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import os

from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.data_loader import DataLoader
from src.embeddings import EmbeddingManager
from src.utils.logger import logger


class RAGPipeline:
    """
    Pipeline completo de RAG para consultas sobre datos de Retail 360.
    Combina recuperación de documentos relevantes con generación de respuestas.
    """
    
    def __init__(self,
                 model_name: str = "llama3",
                 embedding_model: str = "nomic-embed-text",
                 data_path: Optional[str] = None,
                 vector_store_path: str = "./data/vectorstore",
                 vector_store_type: str = "chroma",
                 temperature: float = 0.1,
                 top_k: int = 5,
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Inicializa el pipeline RAG.
        
        Args:
            model_name: Modelo de Ollama para generación
            embedding_model: Modelo para embeddings
            data_path: Ruta al archivo Excel con datos
            vector_store_path: Ruta del vector store
            vector_store_type: Tipo de vector store ('chroma' o 'faiss')
            temperature: Temperatura del modelo (0.0 = determinista, 1.0 = creativo)
            top_k: Número de documentos a recuperar
            ollama_base_url: URL de Ollama
        """
        self.model_name = model_name
        self.data_path = data_path
        self.temperature = temperature
        self.top_k = top_k
        
        logger.info(f"Inicializando RAG Pipeline con modelo: {model_name}")
        
        # Inicializar LLM
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url=ollama_base_url
        )
        
        # Inicializar gestor de embeddings
        self.embedding_manager = EmbeddingManager(
            embedding_model=embedding_model,
            vector_store_type=vector_store_type,
            vector_store_path=vector_store_path,
            ollama_base_url=ollama_base_url
        )
        
        self.qa_chain = None
        self.is_initialized = False
    
    def initialize_from_data(self, 
                           sheet_names: Optional[List[str]] = None,
                           force_recreate: bool = False) -> None:
        """
        Inicializa el pipeline cargando datos desde Excel.
        
        Args:
            sheet_names: Hojas específicas a cargar (None = todas)
            force_recreate: Si True, recrea el vector store
        """
        if not self.data_path:
            raise ValueError("Debe proporcionar data_path al inicializar")
        
        logger.info("=== Inicializando pipeline desde datos ===")
        
        # 1. Cargar datos
        logger.info("Paso 1/3: Cargando datos desde Excel...")
        loader = DataLoader(self.data_path)
        loader.load_excel(sheet_names)
        
        # 2. Crear documentos
        logger.info("Paso 2/3: Creando documentos...")
        documents = loader.create_documents(format_style="detailed")
        summary_docs = loader.get_summary_documents()
        all_documents = documents + summary_docs
        
        logger.info(f"Total de documentos: {len(all_documents)}")
        
        # 3. Crear vector store
        logger.info("Paso 3/3: Creando vector store...")
        self.embedding_manager.create_vector_store(
            all_documents,
            force_recreate=force_recreate
        )
        
        # 4. Crear cadena QA
        self._create_qa_chain()
        
        self.is_initialized = True
        logger.info("✓ Pipeline inicializado exitosamente")
    
    def initialize_from_existing_store(self, collection_name: str = "retail360") -> None:
        """
        Inicializa el pipeline cargando un vector store existente.
        
        Args:
            collection_name: Nombre de la colección (para Chroma)
        """
        logger.info("=== Inicializando pipeline desde vector store existente ===")
        
        # Cargar vector store
        self.embedding_manager.load_vector_store(collection_name)
        
        # Crear cadena QA
        self._create_qa_chain()
        
        self.is_initialized = True
        logger.info("✓ Pipeline inicializado exitosamente")
    
    def _create_qa_chain(self) -> None:
        """Crea la cadena de pregunta-respuesta"""
        
        # Crear retriever
        retriever = self.embedding_manager.get_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # Prompt template personalizado
        template = """Eres un asistente inteligente para Retail 360, una cadena regional de tiendas.
Tu trabajo es responder preguntas sobre los datos de negocio de la empresa basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS IMPORTANTES:
1. Responde SOLO basándote en el contexto proporcionado
2. Si no encuentras la información en el contexto, di "No tengo información suficiente para responder esa pregunta"
3. No inventes datos ni hagas suposiciones
4. Sé preciso y conciso en tus respuestas
5. Si mencionas cifras, asegúrate de que estén en el contexto
6. Responde en español

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Función para formatear documentos
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Crear cadena usando LCEL (LangChain Expression Language)
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Cadena QA creada exitosamente")
    
    def query(self, question: str, return_sources: bool = False) -> Dict[str, Any]:
        """
        Realiza una consulta al pipeline RAG.
        
        Args:
            question: Pregunta en lenguaje natural
            return_sources: Si True, incluye los documentos fuente
        
        Returns:
            Diccionario con la respuesta y opcionalmente las fuentes
        """
        if not self.is_initialized:
            raise RuntimeError("El pipeline no está inicializado. Llame a initialize_from_data() o initialize_from_existing_store()")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CONSULTA: {question}")
        logger.info(f"{'='*60}")
        
        # Ejecutar consulta
        answer = self.qa_chain.invoke(question)
        
        response = {
            "question": question,
            "answer": answer,
        }
        
        if return_sources:
            # Obtener documentos fuente por separado
            docs = self.get_relevant_documents(question)
            response["source_documents"] = docs
            logger.info(f"\nDocumentos recuperados: {len(docs)}")
        
        return response
    
    def query_simple(self, question: str) -> str:
        """
        Consulta simple que solo retorna la respuesta como string.
        
        Args:
            question: Pregunta en lenguaje natural
        
        Returns:
            Respuesta en texto
        """
        result = self.query(question, return_sources=False)
        return result["answer"]
    
    def get_relevant_documents(self, question: str, k: Optional[int] = None) -> List[Document]:
        """
        Obtiene los documentos más relevantes para una consulta sin generar respuesta.
        
        Args:
            question: Pregunta en lenguaje natural
            k: Número de documentos (None = usar top_k por defecto)
        
        Returns:
            Lista de documentos relevantes
        """
        if not self.is_initialized:
            raise RuntimeError("El pipeline no está inicializado")
        
        k = k or self.top_k
        return self.embedding_manager.similarity_search(question, k=k)
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con Ollama y los embeddings.
        
        Returns:
            True si todo funciona correctamente
        """
        logger.info("=== Probando conexión con Ollama ===")
        
        try:
            # Probar LLM
            logger.info("Probando generación de texto...")
            response = self.llm.invoke("Responde con 'OK' si recibes este mensaje")
            logger.info(f"✓ LLM respondió: {response[:100]}...")
            
            # Probar embeddings
            embeddings_ok = self.embedding_manager.test_embeddings()
            
            if embeddings_ok:
                logger.info("✓ Todas las conexiones funcionan correctamente")
                return True
            else:
                logger.error("✗ Error en embeddings")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error en conexión: {e}")
            return False
    
    def update_temperature(self, temperature: float) -> None:
        """
        Actualiza la temperatura del modelo.
        
        Args:
            temperature: Nueva temperatura (0.0 - 1.0)
        """
        self.temperature = temperature
        self.llm.temperature = temperature
        logger.info(f"Temperatura actualizada a: {temperature}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del pipeline.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "is_initialized": self.is_initialized,
            "vector_store_type": self.embedding_manager.vector_store_type,
        }
        
        return stats


# Función de conveniencia para uso rápido
def create_rag_pipeline(data_path: str,
                       model_name: str = "llama3",
                       force_recreate: bool = False) -> RAGPipeline:
    """
    Crea e inicializa un pipeline RAG completo en un solo paso.
    
    Args:
        data_path: Ruta al archivo Excel
        model_name: Modelo de Ollama a usar
        force_recreate: Si True, recrea el vector store
    
    Returns:
        Pipeline RAG inicializado
    """
    pipeline = RAGPipeline(
        model_name=model_name,
        data_path=data_path
    )
    
    pipeline.initialize_from_data(force_recreate=force_recreate)
    
    return pipeline
