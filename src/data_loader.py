"""
Cargador y procesador de datos desde Excel para Retail 360.
Este módulo se encarga de leer el archivo Excel con datos de Power BI
y convertirlos en documentos estructurados para el RAG.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from src.utils.logger import logger


@dataclass
class DatasetInfo:
    """Información sobre el dataset cargado"""
    total_records: int
    sheets: List[str]
    date_range: Optional[tuple] = None
    summary: Dict[str, Any] = None


class DataLoader:
    """
    Clase para cargar y procesar datos desde Excel.
    Convierte los datos en documentos estructurados para el RAG.
    """
    
    def __init__(self, data_path: str):
        """
        Inicializa el cargador de datos.
        
        Args:
            data_path: Ruta al archivo Excel con los datos
        """
        self.data_path = Path(data_path)
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.dataset_info: Optional[DatasetInfo] = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.data_path}")
        
        logger.info(f"DataLoader inicializado con: {self.data_path}")
    
    def load_excel(self, sheet_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Carga el archivo Excel y retorna un diccionario de DataFrames.
        
        Args:
            sheet_names: Lista de hojas a cargar. Si es None, carga todas.
        
        Returns:
            Diccionario con {nombre_hoja: DataFrame}
        """
        logger.info(f"Cargando archivo Excel: {self.data_path}")
        
        try:
            if sheet_names:
                self.dataframes = {
                    sheet: pd.read_excel(self.data_path, sheet_name=sheet)
                    for sheet in sheet_names
                }
            else:
                # Cargar todas las hojas
                excel_file = pd.ExcelFile(self.data_path)
                self.dataframes = {
                    sheet: pd.read_excel(excel_file, sheet_name=sheet)
                    for sheet in excel_file.sheet_names
                }
            
            logger.info(f"Hojas cargadas: {list(self.dataframes.keys())}")
            self._generate_dataset_info()
            
            return self.dataframes
            
        except Exception as e:
            logger.error(f"Error al cargar Excel: {e}")
            raise
    
    def _generate_dataset_info(self) -> None:
        """Genera información resumida del dataset"""
        total_records = sum(len(df) for df in self.dataframes.values())
        sheets = list(self.dataframes.keys())
        
        summary = {}
        for sheet_name, df in self.dataframes.items():
            summary[sheet_name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict()
            }
        
        self.dataset_info = DatasetInfo(
            total_records=total_records,
            sheets=sheets,
            summary=summary
        )
        
        logger.info(f"Dataset info: {total_records} registros en {len(sheets)} hojas")
    
    def create_documents(self, 
                        include_metadata: bool = True,
                        format_style: str = "detailed") -> List[Document]:
        """
        Convierte los DataFrames en documentos de LangChain.
        
        Args:
            include_metadata: Si incluir metadatos en los documentos
            format_style: Estilo de formateo ('detailed', 'compact', 'json')
        
        Returns:
            Lista de objetos Document
        """
        if not self.dataframes:
            raise ValueError("Primero debe cargar los datos con load_excel()")
        
        logger.info("Creando documentos para RAG...")
        documents = []
        
        for sheet_name, df in self.dataframes.items():
            logger.info(f"Procesando hoja: {sheet_name} ({len(df)} registros)")
            
            # Crear un documento por cada fila
            for idx, row in df.iterrows():
                content = self._format_row(row, sheet_name, format_style)
                
                metadata = {
                    "source": str(self.data_path),
                    "sheet": sheet_name,
                    "row_index": idx,
                } if include_metadata else {}
                
                # Agregar columnas importantes al metadata
                if include_metadata:
                    for col in df.columns:
                        if pd.notna(row[col]):
                            # Convertir a tipos serializables
                            value = row[col]
                            if isinstance(value, (pd.Timestamp, datetime)):
                                value = value.isoformat()
                            elif isinstance(value, (int, float, str, bool)):
                                metadata[col] = value
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
        
        logger.info(f"Total de documentos creados: {len(documents)}")
        return documents
    
    def _format_row(self, row: pd.Series, sheet_name: str, style: str = "detailed") -> str:
        """
        Formatea una fila como texto estructurado.
        
        Args:
            row: Serie de pandas con los datos
            sheet_name: Nombre de la hoja
            style: Estilo de formateo
        
        Returns:
            Texto formateado
        """
        if style == "compact":
            # Formato compacto: key=value separado por comas
            items = [f"{k}={v}" for k, v in row.items() if pd.notna(v)]
            return f"[{sheet_name}] " + ", ".join(items)
        
        elif style == "json":
            # Formato JSON-like
            import json
            data = {k: str(v) for k, v in row.items() if pd.notna(v)}
            return json.dumps({"sheet": sheet_name, "data": data}, ensure_ascii=False)
        
        else:  # detailed (default)
            # Formato detallado y legible
            lines = [f"=== Registro de {sheet_name} ==="]
            for key, value in row.items():
                if pd.notna(value):
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
    
    def get_summary_documents(self) -> List[Document]:
        """
        Crea documentos de resumen para cada hoja del Excel.
        Útil para preguntas generales sobre el dataset.
        
        Returns:
            Lista de documentos de resumen
        """
        if not self.dataframes:
            raise ValueError("Primero debe cargar los datos con load_excel()")
        
        summary_docs = []
        
        for sheet_name, df in self.dataframes.items():
            # Estadísticas básicas
            summary_lines = [
                f"=== Resumen de {sheet_name} ===",
                f"Total de registros: {len(df)}",
                f"Columnas: {', '.join(df.columns)}",
                "",
                "Estadísticas:"
            ]
            
            # Agregar estadísticas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                stats = df[col].describe()
                summary_lines.append(f"\n{col}:")
                summary_lines.append(f"  - Promedio: {stats['mean']:.2f}")
                summary_lines.append(f"  - Mínimo: {stats['min']:.2f}")
                summary_lines.append(f"  - Máximo: {stats['max']:.2f}")
            
            # Valores únicos en columnas categóricas
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols[:5]:  # Limitar a 5 columnas
                unique_count = df[col].nunique()
                summary_lines.append(f"\n{col}: {unique_count} valores únicos")
                if unique_count < 20:
                    summary_lines.append(f"  Valores: {', '.join(map(str, df[col].unique()))}")
            
            content = "\n".join(summary_lines)
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(self.data_path),
                    "sheet": sheet_name,
                    "type": "summary"
                }
            )
            summary_docs.append(doc)
        
        logger.info(f"Documentos de resumen creados: {len(summary_docs)}")
        return summary_docs
    
    def filter_by_date_range(self, 
                            df: pd.DataFrame,
                            date_column: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filtra un DataFrame por rango de fechas.
        
        Args:
            df: DataFrame a filtrar
            date_column: Nombre de la columna de fecha
            start_date: Fecha de inicio (formato: YYYY-MM-DD)
            end_date: Fecha de fin (formato: YYYY-MM-DD)
        
        Returns:
            DataFrame filtrado
        """
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        if start_date:
            df_copy = df_copy[df_copy[date_column] >= pd.to_datetime(start_date)]
        if end_date:
            df_copy = df_copy[df_copy[date_column] <= pd.to_datetime(end_date)]
        
        return df_copy


# Función de utilidad para uso rápido
def load_and_create_documents(data_path: str, 
                              sheet_names: Optional[List[str]] = None,
                              include_summary: bool = True) -> List[Document]:
    """
    Función de conveniencia para cargar datos y crear documentos en un solo paso.
    
    Args:
        data_path: Ruta al archivo Excel
        sheet_names: Hojas a cargar (None = todas)
        include_summary: Si incluir documentos de resumen
    
    Returns:
        Lista de documentos listos para el RAG
    """
    loader = DataLoader(data_path)
    loader.load_excel(sheet_names)
    
    documents = loader.create_documents()
    
    if include_summary:
        summary_docs = loader.get_summary_documents()
        documents.extend(summary_docs)
    
    return documents
