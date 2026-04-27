"""
Etsy Listing Generator API - Para Railway
API endpoints para n8n Cloud
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import xml.etree.ElementTree as ET
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import os
from pathlib import Path

app = FastAPI(title="Etsy Listing Generator API")

# CORS para n8n Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB setup
CHROMA_PATH = "/data/presets-vectordb"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Models
class SearchPresetsRequest(BaseModel):
    xmp_content: str  # Contenido del XMP como string
    num_results: int = 200

class ConvertXMPRequest(BaseModel):
    variaciones: list
    tema: str

class AddPresetsRequest(BaseModel):
    presets: List[dict]  # Lista de XMP ya parseados

# ============================================
# ENDPOINT 1: Health Check
# ============================================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Etsy Listing Generator API",
        "version": "1.0",
        "endpoints": [
            "/search-presets",
            "/convert-to-xmp",
            "/add-presets",
            "/health"
        ]
    }

@app.get("/health")
async def health():
    """Verificar que la API funciona"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(
            name="lightroom_presets",
            embedding_function=embedding_function
        )
        total_presets = collection.count()
        
        return {
            "status": "healthy",
            "chromadb": "connected",
            "total_presets": total_presets
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

# ============================================
# ENDPOINT 2: Search Similar Presets
# ============================================
def parse_xmp_content(xmp_content: str) -> dict:
    """Parsea contenido XMP y extrae parámetros"""
    try:
        root = ET.fromstring(xmp_content)
        
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/'
        }
        
        desc = root.find('.//rdf:Description', ns)
        if desc is None:
            return {}
        
        parametros = {}
        for attr, value in desc.attrib.items():
            if attr.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                param_name = attr.replace('{http://ns.adobe.com/camera-raw-settings/1.0/}', '')
                try:
                    if '.' in str(value):
                        parametros[param_name] = float(value)
                    else:
                        parametros[param_name] = int(value)
                except (ValueError, AttributeError):
                    parametros[param_name] = value
        
        return parametros
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parseando XMP: {str(e)}")

@app.post("/search-presets")
async def search_presets(request: SearchPresetsRequest):
    """
    Busca presets similares al XMP base usando ChromaDB
    """
    try:
        # Parsear XMP
        parametros_base = parse_xmp_content(request.xmp_content)
        
        if not parametros_base:
            raise HTTPException(status_code=400, detail="No se pudieron extraer parámetros del XMP")
        
        # Conectar a ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(
            name="lightroom_presets",
            embedding_function=embedding_function
        )
        
        # Documento de consulta
        query_doc = json.dumps(parametros_base, sort_keys=True)
        
        # Buscar similares
        results = collection.query(
            query_texts=[query_doc],
            n_results=min(request.num_results, collection.count())
        )
        
        # Formatear resultados
        similar_presets = []
        
        if results and 'documents' in results and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                preset_params = json.loads(results['documents'][0][i])
                
                preset_info = {
                    "id": results['ids'][0][i],
                    "nombre": results['metadatas'][0][i]['nombre'],
                    "categoria": results['metadatas'][0][i].get('categoria', 'general'),
                    "parametros": preset_params,
                    "distancia": results['distances'][0][i] if 'distances' in results else None
                }
                similar_presets.append(preset_info)
        
        return {
            "success": True,
            "xmp_base_parametros": parametros_base,
            "total_presets_en_db": collection.count(),
            "resultados_solicitados": request.num_results,
            "resultados_encontrados": len(similar_presets),
            "presets_similares": similar_presets
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")

# ============================================
# ENDPOINT 3: Convert JSON to XMP
# ============================================
XMP_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
{parameters}
   crs:Version="15.4"
   crs:ProcessVersion="11.0"
   crs:WhiteBalance="As Shot"
   crs:HasSettings="True"
   crs:HasCrop="False"
   crs:AlreadyApplied="False">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>'''

def format_parameter(key: str, value) -> str:
    """Formatea un parámetro para XMP"""
    # Mapeo de nombres
    param_map = {
        'exposure': 'Exposure2012',
        'contrast': 'Contrast2012',
        'highlights': 'Highlights2012',
        'shadows': 'Shadows2012',
        'whites': 'Whites2012',
        'blacks': 'Blacks2012',
        'clarity': 'Clarity2012',
        'vibrance': 'Vibrance',
        'saturation': 'Saturation',
        'temperature': 'Temperature',
        'tint': 'Tint',
    }
    
    # HSL colors
    hsl_colors = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']
    for color in hsl_colors:
        param_map[f'hue_{color.lower()}'] = f'HueAdjustment{color}'
        param_map[f'saturation_{color.lower()}'] = f'SaturationAdjustment{color}'
        param_map[f'luminance_{color.lower()}'] = f'LuminanceAdjustment{color}'
    
    param_name = param_map.get(key.lower(), key)
    
    if key[0].isupper() or key.startswith('crs:'):
        param_name = key.replace('crs:', '')
    
    if isinstance(value, bool):
        value_str = 'True' if value else 'False'
    elif isinstance(value, (int, float)):
        value_str = str(value)
    else:
        value_str = str(value)
    
    return f'   crs:{param_name}="{value_str}"'

@app.post("/convert-to-xmp")
async def convert_to_xmp(request: ConvertXMPRequest):
    """
    Convierte JSON de variaciones a archivos XMP
    Devuelve los XMP como strings (n8n los guardará)
    """
    try:
        xmp_files = []
        
        for i, variacion in enumerate(request.variaciones, 1):
            if 'parametros' not in variacion:
                continue
            
            parametros = variacion['parametros']
            
            # Formatear parámetros
            param_lines = []
            for key, value in parametros.items():
                param_lines.append(format_parameter(key, value))
            
            parameters_str = '\n'.join(param_lines)
            xmp_content = XMP_TEMPLATE.format(parameters=parameters_str)
            
            filename = f"{request.tema} - Joan's Presets-{i:02d}.xmp"
            
            xmp_files.append({
                "filename": filename,
                "content": xmp_content,
                "descripcion": variacion.get('descripcion', ''),
                "tipo": variacion.get('tipo', '')
            })
        
        return {
            "success": True,
            "total_archivos": len(xmp_files),
            "archivos": xmp_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error convirtiendo a XMP: {str(e)}")

# ============================================
# ENDPOINT 4: Add New Presets
# ============================================
@app.post("/add-presets")
async def add_presets(files: List[UploadFile] = File(...)):
    """
    Añade nuevos presets XMP a ChromaDB
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            collection = client.get_collection(
                name="lightroom_presets",
                embedding_function=embedding_function
            )
        except:
            collection = client.create_collection(
                name="lightroom_presets",
                embedding_function=embedding_function
            )
        
        current_count = collection.count()
        presets_añadidos = 0
        presets_fallidos = 0
        
        for file in files:
            if not file.filename.endswith('.xmp'):
                continue
            
            # Leer contenido
            content = await file.read()
            xmp_content = content.decode('utf-8')
            
            # Parsear
            parametros = parse_xmp_content(xmp_content)
            
            if not parametros:
                presets_fallidos += 1
                continue
            
            # Detectar categoría
            preset_nombre = Path(file.filename).stem
            categoria = "general"
            nombre_lower = preset_nombre.lower()
            
            if any(word in nombre_lower for word in ['forest', 'nature', 'landscape', 'tree']):
                categoria = "nature"
            elif any(word in nombre_lower for word in ['portrait', 'skin', 'face']):
                categoria = "portrait"
            elif any(word in nombre_lower for word in ['urban', 'city', 'street']):
                categoria = "urban"
            elif any(word in nombre_lower for word in ['moody', 'dark', 'cinematic']):
                categoria = "moody"
            
            # ID único
            preset_id = f"preset_{current_count + presets_añadidos + 1:05d}"
            
            # Añadir a ChromaDB
            documento = json.dumps(parametros, sort_keys=True)
            metadata = {
                "nombre": preset_nombre,
                "categoria": categoria,
                "archivo_original": file.filename
            }
            
            try:
                collection.add(
                    documents=[documento],
                    metadatas=[metadata],
                    ids=[preset_id]
                )
                presets_añadidos += 1
            except Exception as e:
                presets_fallidos += 1
        
        return {
            "success": True,
            "presets_añadidos": presets_añadidos,
            "presets_fallidos": presets_fallidos,
            "total_en_db": current_count + presets_añadidos
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error añadiendo presets: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
