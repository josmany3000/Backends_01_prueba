import os
import uuid
import json
import requests
import logging
import time
import re
import threading
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- 1. CONFIGURACIÓN INICIAL Y LOGGING ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Configuración de credenciales de GCP
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
    credentials_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    credentials_path = f'/tmp/{uuid.uuid4()}_gcp-credentials.json'
    try:
        with open(credentials_path, 'w') as f:
            f.write(credentials_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        logging.info("Credenciales de GCP cargadas desde variable de entorno.")
    except Exception as e:
        logging.error("No se pudieron escribir las credenciales de GCP en el archivo temporal.", exc_info=True)

app = Flask(__name__)
CORS(app)

JOBS = {}

# --- Configuración de ElevenLabs ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"

if not ELEVENLABS_API_KEY:
    logging.warning("La API Key de ElevenLabs (ELEVENLABS_API_KEY) no está configurada. La generación de audio fallará.")

# --- Configuración de Clientes de Google ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    storage_client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GCP_REGION", "us-central1"))
    logging.info("Clientes de Google (Gemini, Storage, VertexAI) configurados exitosamente.")
except Exception as e:
    logging.critical("ERROR FATAL AL CONFIGURAR CLIENTES DE GOOGLE.", exc_info=True)
    
model_text = genai.GenerativeModel('gemini-1.5-flash')
model_image = ImageGenerationModel.from_pretrained("imagegeneration@006")

# --- NUEVO: DICCIONARIO DE PROMPTS POR NICHO ---
NICHE_PROMPTS = {
    "misterio_terror": """
    **ROL:** Eres un guionista experto en el género de misterio y terror.
    **TAREA:** Escribe una narración de suspenso y terror sobre el siguiente tema: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** Usa un tono oscuro, misterioso y con giros inesperados. Mantén al oyente al borde del asiento y genera tensión con descripciones visuales y auditivas. Puede ser sobre una leyenda, una historia documentada o un testimonio de la vida real.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "finanzas_emprendimiento": """
    **ROL:** Eres un coach financiero y de negocios, experto en comunicación.
    **TAREA:** Redacta una narración inspiradora sobre el siguiente tema: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** Utiliza un tono motivador, claro y profesional. Incluye datos curiosos, estrategias prácticas y consejos para emprendedores modernos. Puede ser una historia de éxito, un análisis de tendencia o un consejo práctico.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "tecnologia_ia": """
    **ROL:** Eres un divulgador científico y experto en nuevas tecnologías.
    **TAREA:** Genera una narración informativa y futurista sobre este tema: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** El estilo debe ser didáctico, emocionante y accesible para todo público. Investiga brevemente en tu base de conocimiento para asegurar que los datos sean recientes y correctos. Incluye ejemplos reales y una visión de futuro.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "documentales": """
    **ROL:** Eres un guionista de documentales para canales como Discovery o National Geographic.
    **TAREA:** Escribe una narración objetiva, informativa y neutral sobre el siguiente tema: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** El tono debe ser serio y documental, con un enfoque en hechos, fechas y análisis profundos. La información debe ser verificable y presentada de manera clara y ordenada.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "biblia_cristianismo": """
    **ROL:** Eres un predicador y maestro de estudios bíblicos con gran capacidad para narrar.
    **TAREA:** Redacta una narración inspiradora sobre este tema: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** Usa un tono respetuoso, cálido y espiritual. El contenido puede basarse en pasajes bíblicos, reflexiones cristianas o historias de fe. Transmite paz, esperanza y enseñanzas morales con un lenguaje accesible para todos.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "aliens_teorias": """
    **ROL:** Eres un investigador y presentador de programas sobre misterios sin resolver.
    **TAREA:** Crea una narración intrigante sobre este tema: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** Usa un estilo misterioso y especulativo. Puedes hablar de teorías conspirativas o casos famosos de contacto extraterrestre. Utiliza referencias a eventos, lugares o personas reales para dar contexto, pero mantén un tono de entretenimiento y misterio, sin afirmar que la teoría es 100% verídica.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "tendencias_virales": """
    **ROL:** Eres un influencer y creador de contenido experto en tendencias de redes sociales.
    **TAREA:** Genera una narración sobre la siguiente tendencia viral: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** Usa un tono dinámico, moderno y con lenguaje juvenil. El ritmo debe ser divertido, acelerado y llamativo. Puedes incluir hashtags, expresiones virales y el contexto relevante de la tendencia para que todos la entiendan.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "politica": """
    **ROL:** Eres un analista político y periodista de investigación.
    **TAREA:** Redacta una narración seria y crítica sobre el siguiente tema político: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** Usa un tono imparcial pero analítico. Cita hechos, posibles estadísticas y consecuencias. Asegúrate de explicar el contexto de manera clara y con profundidad, sin caer en lenguaje excesivamente técnico. El objetivo es informar, no persuadir.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """,
    "anime_manga": """
    **ROL:** Eres un fan y crítico experto en la cultura del anime y el manga.
    **TAREA:** Crea una narración apasionada sobre este tema: "{guion_personalizado}".
    **INSTRUCCIONES DE ESTILO:** Usa un tono épico, emocional y juvenil. La narración puede ser sobre un anime/manga popular o una historia original inspirada en ese estilo. Incluye referencias al estilo narrativo japonés, con momentos de dramatismo, reflexión y acción.
    **REGLAS TÉCNICAS OBLIGATORIAS:**
    1.  **IDIOMA:** El guion debe estar en **Español Latinoamericano**.
    2.  **ESTRUCTURA:** Genera **EXACTAMENTE {numero_de_escenas} escenas**.
    3.  **LONGITUD:** Cada escena debe tener un máximo de **{palabras_por_escena} palabras**.
    4.  **GUION LIMPIO:** El texto de cada escena debe ser solo la narración. **NO INCLUYAS encabezados (ej. 'ESCENA 1'), nombres de personajes, o cualquier otra etiqueta.**
    5.  **FORMATO DE SALIDA (CRÍTICO):** La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto en el array debe tener las claves "id" y "script".
    """
}

# --- 2. DECORADOR DE REINTENTOS (Sin cambios) ---
def retry_on_failure(retries=3, delay=5, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, IndexError):
                         logging.error(f"Error de IndexError en {func.__name__}. No se reintentará. Causa probable: contenido bloqueado por la API.")
                         raise e
                    logging.warning(f"Intento {i + 1}/{retries} para {func.__name__} falló: {e}. Reintentando en {current_delay}s...")
                    if i == retries - 1:
                        logging.error(f"Todos los {retries} intentos para {func.__name__} fallaron.", exc_info=True)
                        raise e
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

# --- 3. FUNCIONES AUXILIARES (Sin cambios) ---
@retry_on_failure()
def upload_to_gcs(file_stream, destination_blob_name, content_type):
    logging.info(f"Iniciando subida a GCS. Bucket: {GCS_BUCKET_NAME}, Destino: {destination_blob_name}")
    if not GCS_BUCKET_NAME:
        raise ValueError("El nombre del bucket de GCS no está configurado.")
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_stream, content_type=content_type)
    blob.make_public()
    logging.info(f"Subida a GCS exitosa. URL pública: {blob.public_url}")
    return blob.public_url

def safe_json_parse(text):
    text = text.strip().replace('```json', '').replace('```', '')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.error(f"Error al decodificar JSON. Texto problemático: {text[:500]}", exc_info=True)
        return None

@retry_on_failure()
def _get_keywords_for_image_prompt(script_text):
    prompt = f"""
    Analiza el texto de la escena. Extrae 4-5 palabras clave para un generador de imágenes.
    **REGLAS CRÍTICAS DE SEGURIDAD:**
    1.  **MÁXIMA SEGURIDAD:** Las palabras clave deben ser 100% seguras, neutrales e inofensivas.
    2.  **EVITAR PERSONAS:** No incluyas palabras clave que describan personas, partes del cuerpo o edad (ej. 'niño', 'mujer', 'rostro', 'piel').
    3.  **ENFOQUE EN OBJETOS Y AMBIENTES:** Céntrate EXCLUSIVAMENTE en objetos inanimados, lugares, ambientes y conceptos abstractos (ej. 'laboratorio', 'microscopio', 'galaxia lejana', 'mapa antiguo', 'misterio', 'tecnología').
    4.  **FORMATO ESTRICTO:** Devuelve únicamente las palabras clave en español, separadas por comas. NADA MÁS.

    **Texto de la Escena:**
    ---
    {script_text}
    ---
    """
    response = model_text.generate_content(prompt)
    keywords = response.text.strip().replace("`", "")
    logging.info(f"Keywords de seguridad generadas para el guion: '{keywords}'")
    return keywords

@retry_on_failure()
def _generate_and_upload_image(scene_script, aspect_ratio):
    keywords = _get_keywords_for_image_prompt(scene_script)
    logging.info(f"Generando imagen desde keywords de seguridad: '{keywords}' con aspect ratio: {aspect_ratio}")
    image_prompt = f"cinematic still, photorealistic, high detail of: {keywords}"
    
    try:
        images = model_image.generate_images(
            prompt=image_prompt,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            negative_prompt="text, watermark, logo, blurry, words, letters, signature, person, people, face, skin"
        )
        if not images:
            logging.warning(f"La API no devolvió imágenes para las keywords: '{keywords}'. La solicitud pudo ser bloqueada. Retornando None.")
            return None 
        
        public_gcs_url = upload_to_gcs(images[0]._image_bytes, f"images/img_{uuid.uuid4()}.png", 'image/png')
        return public_gcs_url
    except Exception as e:
        logging.error(f"Excepción durante la llamada a generate_images para '{keywords}': {e}", exc_info=True)
        return None

@retry_on_failure()
def _generate_audio_with_elevenlabs(text_input, voice_id):
    logging.info(f"Llamando a la API de ElevenLabs con la voz '{voice_id}'.")
    if not ELEVENLABS_API_KEY:
        raise ValueError("La API Key de ElevenLabs no está configurada.")
    tts_url = f"{ELEVENLABS_API_URL}/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    data = {"text": text_input, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    response = requests.post(tts_url, json=data, headers=headers)
    response.raise_for_status() 
    logging.info("Respuesta de la API de ElevenLabs recibida exitosamente.")
    public_url = upload_to_gcs(response.content, f"audio/audio_{uuid.uuid4()}.mp3", 'audio/mpeg')
    return public_url


# --- 4. TRABAJADOR DE FONDO (Sin cambios) ---
def _perform_image_generation(job_id, scenes, aspect_ratio):
    total_scenes = len(scenes)
    scenes_con_media = []
    try:
        for i, scene in enumerate(scenes):
            JOBS[job_id]['status'] = 'processing'
            JOBS[job_id]['progress'] = f"{i + 1}/{total_scenes}"
            logging.info(f"Trabajo {job_id}: Procesando imagen {i+1}/{total_scenes}")
            scene['id'] = scene.get('id', f'scene-{uuid.uuid4()}')
            
            image_url = _generate_and_upload_image(scene['script'], aspect_ratio)
            
            if image_url:
                scene['imageUrl'] = image_url
            else:
                logging.error(f"Trabajo {job_id}: Fallo definitivo al generar imagen para la escena {scene['id']}. Se usará un placeholder.")
                error_img_res = '1080x1920' if aspect_ratio == '9:16' else '1920x1080'
                scene['imageUrl'] = f"https://via.placeholder.com/{error_img_res}?text=Error+Al+Generar"
            
            scene['videoUrl'] = None
            scenes_con_media.append(scene)
            
            if i < total_scenes - 1:
                logging.info(f"Trabajo {job_id}: Pausando por 5 segundos.")
                time.sleep(5)

        JOBS[job_id]['status'] = 'completed'
        JOBS[job_id]['result'] = {"scenes": scenes_con_media}
        logging.info(f"Trabajo {job_id} completado exitosamente.")
    except Exception as e:
        logging.error(f"Trabajo {job_id} falló catastróficamente: {e}", exc_info=True)
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)


# --- 5. ENDPOINTS DE LA API ---
@app.route("/")
def index():
    return "Backend de IA para Videos v5.3 - Audio con ElevenLabs y Prompts por Nicho"

# --- ENDPOINT DE GENERACIÓN INICIAL TOTALMENTE MODIFICADO ---
@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    try:
        data = request.get_json()
        logging.info(f"Recibida solicitud de trabajo para generar contenido con datos: {data}")

        # 1. Obtener datos de la solicitud con valores por defecto
        # El nicho por defecto
