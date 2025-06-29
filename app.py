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

# --- DICCIONARIO CENTRAL DE PROMPTS POR NICHO ---
PROMPTS_POR_NICHO = {
    "misterio_terror": "**GANCHO INICIAL OBLIGATORIO:** Comienza la narración con una pregunta intrigante que enganche al usuario, como por ejemplo: '¿Sabías que...?', '¿Te has preguntado alguna vez...?' o '¿Qué pasaría si te dijera que...?'. A continuación, escribe una narración de suspenso y terror sobre un evento inexplicable, ya sea una leyenda o una historia documentada. Usa un tono oscuro, misterioso y con giros inesperados. Mantén al oyente al borde del asiento y genera tensión con descripciones visuales y auditivas.",
    "finanzas_emprendimiento": "Redacta una narración inspiradora sobre una historia de éxito financiero o de emprendimiento, o un tema financiero que esté en tendencias. Utiliza un tono motivador, claro y profesional. Incluye datos curiosos, estrategias prácticas y consejos para emprendedores modernos.",
    "tecnologia_ia": "Genera una narración informativa y futurista sobre un avance reciente en inteligencia artificial o tecnología disruptiva. Investiga en sitios oficiales. El estilo debe ser didáctico, emocionante y accesible para todo público, con ejemplos reales y visión de futuro.",
    "documentales": "Escribe una narración objetiva, informativa y neutral sobre un tema de interés social, cultural o histórico. El tono debe ser serio y documental, con un enfoque en hechos, fechas y análisis profundos. Ideal para un documental narrado.",
    "biblia_cristianismo": "Redacta una narración inspiradora basada en pasajes bíblicos, reflexiones cristianas o historias de fe. Usa un tono respetuoso, cálido y espiritual. Transmite paz, esperanza y enseñanzas morales con lenguaje accesible.",
    "aliens_teorias": "Crea una narración intrigante sobre una teoría conspirativa o un caso famoso de contacto extraterrestre. Usa un estilo misterioso, especulativo y con referencias reales, manteniendo el tono entretenido pero sin afirmar que es 100% verdad.",
    "tendencias_virales": "Genera una narración dinámica, moderna y con lenguaje juvenil sobre una tendencia viral actual en redes sociales. Usa un tono divertido, acelerado y llamativo. Incluye hashtags, expresiones virales y contexto relevante.",
    "politica": "Redacta una narración seria y crítica sobre un tema político actual. Usa un tono imparcial pero analítico, citando hechos, estadísticas y consecuencias. Asegúrate de explicar el contexto de manera clara y con profundidad.",
    "anime_manga": "Crea una narración apasionada sobre un anime o manga popular o una historia original inspirada en ese estilo. Usa un tono épico, emocional y juvenil. Incluye referencias al estilo narrativo japonés, con dramatismo y acción."
}

# --- 2. DECORADOR DE REINTENTOS ---
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

# --- 3. FUNCIONES AUXILIARES ---
@retry_on_failure()
def upload_to_gcs(file_stream, destination_blob_name, content_type):
    if not GCS_BUCKET_NAME:
        raise ValueError("El nombre del bucket de GCS no está configurado.")
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_stream, content_type=content_type)
    blob.make_public()
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
    prompt_template = "Analiza el texto y extrae 4-5 palabras clave seguras (objetos, lugares) para un generador de imágenes. Formato: palabras separadas por comas. Texto: {script}"
    prompt = prompt_template.format(script=script_text)
    response = model_text.generate_content(prompt)
    return response.text.strip().replace("`", "")

@retry_on_failure()
def _generate_and_upload_image(scene_script, aspect_ratio):
    try:
        keywords = _get_keywords_for_image_prompt(scene_script)
        image_prompt = f"cinematic still, photorealistic, high detail of: {keywords}"
        images = model_image.generate_images(
            prompt=image_prompt,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            negative_prompt="text, watermark, logo, person, people, face, skin"
        )
        if not images:
            return None 
        return upload_to_gcs(images[0]._image_bytes, f"images/img_{uuid.uuid4()}.png", 'image/png')
    except Exception as e:
        logging.error(f"Excepción durante la generación de imagen: {e}", exc_info=True)
        return None

@retry_on_failure()
def _generate_audio_with_elevenlabs(text_input, voice_id):
    if not ELEVENLABS_API_KEY:
        raise ValueError("La API Key de ElevenLabs no está configurada.")
    tts_url = f"{ELEVENLABS_API_URL}/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    data = {"text": text_input, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    response = requests.post(tts_url, json=data, headers=headers)
    response.raise_for_status()
    return upload_to_gcs(response.content, f"audio/audio_{uuid.uuid4()}.mp3", 'audio/mpeg')

# --- 4. TRABAJADOR DE FONDO ---
def _perform_initial_media_generation(job_id, scenes):
    try:
        total_scenes = len(scenes)
        scenes_con_media = []
        for i, scene in enumerate(scenes):
            JOBS[job_id]['progress'] = f"{i + 1}/{total_scenes}"
            scene['id'] = scene.get('id', f'scene-{uuid.uuid4()}')
            scene['imageUrl'] = None
            scene['videoUrl'] = None
            scenes_con_media.append(scene)
        JOBS[job_id]['status'] = 'completed'
        JOBS[job_id]['result'] = {"scenes": scenes_con_media}
        logging.info(f"Trabajo {job_id}: Estructuración de guion completada.")
    except Exception as e:
        logging.error(f"Trabajo {job_id} falló catastróficamente: {e}", exc_info=True)
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)

# --- 5. ENDPOINTS DE LA API ---
@app.route("/")
def index():
    return "Backend de IA para Videos v7.2 - Estable"

@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    try:
        data = request.get_json()
        nicho = data.get('nicho', 'documentales')
        userInput = data.get('userInput')
        tipoEntrada = data.get('tipoEntrada', 'tema')
        idioma = data.get('idioma', 'Español Latinoamericano')
        if not userInput or not userInput.strip():
            return jsonify({"error": "El campo de tema o guion no puede estar vacío."}), 400
        duracion_a_escenas = {"50": 4, "120": 6, "180": 8, "300": 10, "600": 15}
        numero_de_escenas = duracion_a_escenas.get(str(data.get('duracionVideo', '50')), 4)
        prompt_final = ""
        if tipoEntrada == 'guion':
            logging.info("Modo 'Guion Personalizado' detectado.")
            prompt_template_guion = """
            ROL: Eres un editor de video y guionista experto.
            TAREA: Analiza el guion del usuario y reestructúralo en {num_escenas} escenas para un video corto en {idioma}.
            REGLAS:
            1. Divide el guion en EXACTAMENTE {num_escenas} escenas lógicas.
            2. La última escena DEBE tener un llamado a la acción (CTA). Si no lo tiene, agrégale uno apropiado para el nicho '{nicho}'.
            3. El resultado debe ser solo la narración, sin etiquetas como 'ESCENA 1'.
            GUION DEL USUARIO: "{guion_usuario}"
            FORMATO DE SALIDA OBLIGATORIO: Un JSON válido con una clave "scenes", que es un array de objetos. Cada objeto tiene "id" y "script". No incluyas nada más.
            """
            prompt_final = prompt_template_guion.format(num_escenas=numero_de_escenas, idioma=idioma, nicho=nicho, guion_usuario=userInput)
        else:
            logging.info("Modo 'Tema Principal' detectado.")
            instruccion_base = PROMPTS_POR_NICHO.get(nicho, PROMPTS_POR_NICHO['documentales'])
            palabras_totales = int(data.get('duracionVideo', 50)) * 2.8
            palabras_por_escena = int(palabras_totales // numero_de_escenas)
            prompt_template_tema = """
            ROL: Eres un guionista experto para el nicho '{nicho}'.
            TAREA: Crea un guion sobre "{tema_principal}" en {idioma}.
            REGLAS:
            1. {instruccion_base}
            2. Genera EXACTAMENTE {num_escenas} escenas. Cada una con aprox. {num_palabras} palabras.
            3. La última escena DEBE tener un llamado a la acción (CTA).
            4. El guion debe ser solo texto narrativo, sin etiquetas.
            FORMATO DE SALIDA OBLIGATORIO: Un JSON válido con una clave "scenes", que es un array de objetos. Cada objeto tiene "id" y "script". No incluyas nada más.
            """
            prompt_final = prompt_template_tema.format(nicho=nicho, tema_principal=userInput, idioma=idioma, instruccion_base=instruccion_base, num_escenas=numero_de_escenas, num_palabras=palabras_por_escena)
        response = model_text.generate_content(prompt_final)
        parsed_json = safe_json_parse(response.text)
        if not (parsed_json and 'scenes' in parsed_json and isinstance(parsed_json['scenes'], list) and parsed_json['scenes']):
            logging.error(f"La IA no pudo generar un guion JSON válido. Respuesta: {response.text}")
            return jsonify({"error": "La IA no pudo generar un guion válido. Intenta ajustar tu entrada."}), 500
        scenes = parsed_json['scenes']
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {'status': 'pending', 'progress': f'0/{len(scenes)}'}
        thread = threading.Thread(target=_perform_initial_media_generation, args=(job_id, scenes))
        thread.start()
        return jsonify({"jobId": job_id})
    except Exception as e:
        logging.error("Error inesperado en generate_initial_content.", exc_info=True)
        return jsonify({"error": f"Ocurrió un error interno: {e}"}), 500

@app.route('/api/content-job-status/<job_id>', methods=['GET'])
def get_content_job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Trabajo no encontrado"}), 404
    return jsonify(job)

@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
    try:
        data = request.get_json()
        scene = data.get('scene')
        part_to_regenerate = data.get('part')
        config = data.get('config')
        if not all([scene, part_to_regenerate, config]):
            return jsonify({"error": "Faltan datos en la solicitud"}), 400
        if part_to_regenerate == 'script':
            prompt = "Eres un guionista. Reescribe este texto de forma creativa y concisa, en Español Latinoamericano: '{script}'".format(script=scene.get('script'))
            response = model_text.generate_content(prompt)
            return jsonify({"newScript": response.text.strip().replace("`", "")})
        elif part_to_regenerate == 'media':
            aspect_ratio = config.get('resolucion', '16:9')
            new_image_url = _generate_and_upload_image(scene.get('script', 'una imagen abstracta'), aspect_ratio)
            if not new_image_url:
                return jsonify({"error": "La IA no pudo generar una nueva imagen."}), 500
            return jsonify({"newImageUrl": new_image_url, "newVideoUrl": None})
    except Exception as e:
        logging.error(f"Error al regenerar parte de escena: {e}", exc_info=True)
        return jsonify({"error": f"Error interno al regenerar: {str(e)}"}), 500
    return jsonify({"error": "Parte no válida para regenerar."}), 400

@app.route('/api/generate-full-audio', methods=['POST'])
def generate_full_audio():
    try:
        data = request.get_json()
        script = data.get('script')
        voice_id = data.get('voice', '21m00Tcm4TlvDq8ikWAM')
        if not script or not script.strip():
            return jsonify({"error": "El guion es requerido"}), 400
        public_url = _generate_audio_with_elevenlabs(script, voice_id)
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        logging.error(f"Error en generate_full_audio: {e}", exc_info=True)
        return jsonify({"error": f"No se pudo generar el audio: {str(e)}"}), 500

@app.route('/api/voice-sample', methods=['POST'])
def generate_voice_sample():
    try:
        data = request.get_json()
        voice_id = data.get('voice')
        if not voice_id:
            return jsonify({"error": "Se requiere un ID de voz"}), 400
        sample_text = "Hola, esta es una prueba de la voz seleccionada para la narración."
        public_url = _generate_audio_with_elevenlabs(sample_text, voice_id)
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        logging.error("Error al generar muestra de voz: %s", e)
        return jsonify({"error": f"No se pudo generar la muestra de voz: {str(e)}"}), 500

# --- 6. EJECUCIÓN DEL SERVIDOR ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
    
