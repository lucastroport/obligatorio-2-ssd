"""
Módulo de embeddings y gestión del vector store.
Maneja la creación de embeddings con Ollama y almacenamiento en Chroma/FAISS.
"""

from typing import List, Optional, Union
from pathlib import Path
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

from src.utils.logger import logger


class EmbeddingManager:
    """
    Gestiona la creación de embeddings y el vector store.
    Soporta Chroma y FAISS como opciones de almacenamiento.
    """
    
    def __init__(self,
                 embedding_model: str = "nomic-embed-text",
                 vector_store_type: str = "chroma",
                 vector_store_path: str = "./data/vectorstore",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Inicializa el gestor de embeddings.
        
        Args:
            embedding_model: Modelo de Ollama para embeddings
            vector_store_type: Tipo de vector store ('chroma' o 'faiss')
            vector_store_path: Ruta donde guardar el vector store
            chunk_size: Tamaño de los chunks de texto
            chunk_overlap: Solapamiento entre chunks
            ollama_base_url: URL base de Ollama
        """
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type.lower()
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_base_url = ollama_base_url
        
        logger.info(f"Inicializando EmbeddingManager con modelo: {embedding_model}")
        
        # Inicializar embeddings de Ollama
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )
        
        # Text splitter para dividir documentos largos
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_store = None
        
        # Crear directorio si no existe
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos largos en chunks más pequeños.
        
        Args:
            documents: Lista de documentos originales
        
        Returns:
            Lista de documentos divididos en chunks
        """
        logger.info(f"Dividiendo {len(documents)} documentos en chunks...")
        
        split_docs = self.text_splitter.split_documents(documents)
        
        logger.info(f"Documentos divididos en {len(split_docs)} chunks")
        return split_docs
    
    def create_vector_store(self, 
                           documents: List[Document],
                           collection_name: str = "retail360",
                           force_recreate: bool = False) -> Union[Chroma, FAISS]:
        """
        Crea el vector store a partir de los documentos.
        
        Args:
            documents: Lista de documentos a indexar
            collection_name: Nombre de la colección (solo para Chroma)
            force_recreate: Si True, recrea el store aunque ya exista
        
        Returns:
            Instancia del vector store creado
        """
        logger.info(f"Creando vector store tipo: {self.vector_store_type}")
        
        # Dividir documentos en chunks
        split_docs = self.split_documents(documents)
        
        if self.vector_store_type == "chroma":
            return self._create_chroma_store(split_docs, collection_name, force_recreate)
        elif self.vector_store_type == "faiss":
            return self._create_faiss_store(split_docs, force_recreate)
        else:
            raise ValueError(f"Vector store type no soportado: {self.vector_store_type}")
    
    def _create_chroma_store(self, 
                            documents: List[Document],
                            collection_name: str,
                            force_recreate: bool) -> Chroma:
        """Crea un vector store Chroma"""
        persist_directory = str(self.vector_store_path / "chroma")
        
        # Verificar si ya existe
        if Path(persist_directory).exists() and not force_recreate:
            logger.info("Cargando vector store Chroma existente...")
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        else:
            logger.info("Creando nuevo vector store Chroma...")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            logger.info(f"Vector store Chroma guardado en: {persist_directory}")
        
        return self.vector_store
    
    def _create_faiss_store(self,
                           documents: List[Document],
                           force_recreate: bool) -> FAISS:
        """Crea un vector store FAISS"""
        index_path = self.vector_store_path / "faiss" / "index.faiss"
        
        # Verificar si ya existe
        if index_path.exists() and not force_recreate:
            logger.info("Cargando vector store FAISS existente...")
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path / "faiss"),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("Creando nuevo vector store FAISS...")
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Guardar el índice
            save_path = self.vector_store_path / "faiss"
            save_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(save_path))
            logger.info(f"Vector store FAISS guardado en: {save_path}")
        
        return self.vector_store
    
    def load_vector_store(self, 
                         collection_name: str = "retail360") -> Union[Chroma, FAISS]:
        """
        Carga un vector store existente.
        
        Args:
            collection_name: Nombre de la colección (solo para Chroma)
        
        Returns:
            Instancia del vector store cargado
        """
        logger.info(f"Cargando vector store tipo: {self.vector_store_type}")
        
        if self.vector_store_type == "chroma":
            persist_directory = str(self.vector_store_path / "chroma")
            if not Path(persist_directory).exists():
                raise FileNotFoundError(f"No se encontró el vector store en: {persist_directory}")
            
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        
        elif self.vector_store_type == "faiss":
            index_path = self.vector_store_path / "faiss" / "index.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"No se encontró el vector store en: {index_path}")
            
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path / "faiss"),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        else:
            raise ValueError(f"Vector store type no soportado: {self.vector_store_type}")
        
        logger.info("Vector store cargado exitosamente")
        return self.vector_store
    
    def similarity_search(self, 
                         query: str,
                         k: int = 5,
                         filter_dict: Optional[dict] = None) -> List[Document]:
        """
        Realiza búsqueda por similitud en el vector store.
        
        Args:
            query: Consulta en texto
            k: Número de resultados a retornar
            filter_dict: Filtros de metadata (opcional)
        
        Returns:
            Lista de documentos relevantes
        """
        if not self.vector_store:
            raise ValueError("Debe crear o cargar un vector store primero")
        
        logger.info(f"Búsqueda de similitud: '{query}' (top {k})")
        
        if filter_dict:
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        logger.info(f"Encontrados {len(results)} resultados")
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Agrega nuevos documentos al vector store existente.
        
        Args:
            documents: Lista de documentos a agregar
        """
        if not self.vector_store:
            raise ValueError("Debe crear o cargar un vector store primero")
        
        logger.info(f"Agregando {len(documents)} documentos al vector store...")
        
        split_docs = self.split_documents(documents)
        self.vector_store.add_documents(split_docs)
        
        # Persistir cambios (solo para Chroma)
        if self.vector_store_type == "chroma":
            self.vector_store.persist()
        elif self.vector_store_type == "faiss":
            save_path = self.vector_store_path / "faiss"
            self.vector_store.save_local(str(save_path))
        
        logger.info("Documentos agregados exitosamente")
    
    def get_retriever(self, 
                     search_type: str = "similarity",
                     search_kwargs: Optional[dict] = None):
        """
        Obtiene un retriever para usar en cadenas de LangChain.
        
        Args:
            search_type: Tipo de búsqueda ('similarity', 'mmr', 'similarity_score_threshold')
            search_kwargs: Argumentos adicionales para la búsqueda
        
        Returns:
            Retriever configurado
        """
        if not self.vector_store:
            raise ValueError("Debe crear o cargar un vector store primero")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        logger.info(f"Creando retriever con búsqueda tipo: {search_type}")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def test_embeddings(self) -> bool:
        """
        Prueba que los embeddings funcionen correctamente.
        
        Returns:
            True si la prueba es exitosa
        """
        try:
            logger.info("Probando generación de embeddings...")
            test_text = "Esta es una prueba de embeddings para Retail 360"
            embedding = self.embeddings.embed_query(test_text)
            
            logger.info(f"✓ Embedding generado exitosamente (dimensión: {len(embedding)})")
            logger.info(f"  Primeros 5 valores: {embedding[:5]}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error al generar embeddings: {e}")
            return False
