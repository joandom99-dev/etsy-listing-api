#!/usr/bin/env python3
"""
Script para inicializar ChromaDB en Railway
Ejecutar UNA VEZ después del deploy para crear la colección
"""

import chromadb
from chromadb.utils import embedding_functions
import os

CHROMA_PATH = "/data/presets-vectordb"

def init_chromadb():
    print("=" * 60)
    print("INICIALIZANDO CHROMADB EN RAILWAY")
    print("=" * 60)
    print()
    
    # Crear directorio si no existe
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Cliente ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Función de embedding
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Crear colección
    try:
        collection = client.create_collection(
            name="lightroom_presets",
            embedding_function=embedding_function,
            metadata={"description": "Lightroom presets database"}
        )
        print("✅ Colección 'lightroom_presets' creada")
    except Exception as e:
        print(f"⚠️  Colección ya existe o error: {e}")
        collection = client.get_collection(
            name="lightroom_presets",
            embedding_function=embedding_function
        )
    
    print(f"📊 Presets actuales en DB: {collection.count()}")
    print()
    print("=" * 60)
    print("✅ CHROMADB INICIALIZADO")
    print("=" * 60)
    print()
    print("Próximo paso: Subir tus presets XMP usando /add-presets")
    print()

if __name__ == "__main__":
    init_chromadb()
