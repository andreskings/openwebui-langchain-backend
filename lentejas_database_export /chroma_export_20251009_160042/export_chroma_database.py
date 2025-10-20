#!/usr/bin/env python3
import os, json, shutil
from pathlib import Path
import chromadb

def import_chroma_database(import_path, target_path="./chroma_lentejas_v2"):
    print("🌱 Iniciando importación de ChromaDB de LENTEJAS (384D)...")
    
    json_file = Path(import_path) / "chroma_data.json"
    if not json_file.exists():
        return False
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if Path(target_path).exists():
        shutil.rmtree(target_path)
    
    # Cliente simple SIN OpenAI embeddings (igual que trigo)
    client = chromadb.PersistentClient(path=target_path)
    
    for collection_name, collection_data in data['collections'].items():
        print(f"📋 Importando colección: {collection_name}")
        
        # Crear colección SIN embedding function (usa default 384D)
        collection = client.create_collection(name=collection_name)
        
        if collection_data['count'] > 0:
            collection.add(
                documents=collection_data['documents'],
                metadatas=collection_data['metadatas'],
                ids=collection_data['ids']
            )
        print(f"✅ {collection_name}: {collection_data['count']} documentos importados")
    
    print(f"🎉 Importación de LENTEJAS completada: {target_path}")
    return target_path
