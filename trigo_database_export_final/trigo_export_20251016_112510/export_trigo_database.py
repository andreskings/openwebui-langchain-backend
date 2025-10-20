#!/usr/bin/env python3
import os, json, shutil
from pathlib import Path
import chromadb

def import_wheat_database(import_path, target_path="./chroma_trigo_v1"):
    print("🌾 Iniciando importación de ChromaDB de TRIGO...")
    json_file = Path(import_path) / "trigo_data.json"
    if not json_file.exists():
        return False
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if Path(target_path).exists():
        shutil.rmtree(target_path)
    
    client = chromadb.PersistentClient(path=target_path)
    
    for collection_name, collection_data in data['collections'].items():
        print(f"📋 Importando colección de trigo: {collection_name}")
        collection = client.create_collection(name=collection_name)
        
        if collection_data['count'] > 0:
            collection.add(
                documents=collection_data['documents'],
                metadatas=collection_data['metadatas'],
                ids=collection_data['ids']
            )
        print(f"✅ {collection_name}: {collection_data['count']} documentos de trigo importados")
    
    print(f"🎉 Importación de TRIGO completada: {target_path}")
    return target_path
