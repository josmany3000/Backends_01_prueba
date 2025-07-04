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
from vertexai.vision_models import ImageGenerationModel
import redis
from num2words import num2words

# --- 1. CONFIGURACIÓN INICIAL Y VALIDACIÓN ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
    credentials_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    credentials_path = f'/tmp/{uuid.uuid4()}_gcp-credentials.json'
    try:
        with open(credentials_path, 'w') as f:
            f.write(credentials_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        logging.info("Credenciales de GCP cargadas desde variable de entorno.")
    except Exception:
        logging.error("No se pudieron escribir las credenciales de GCP en el archivo temporal.", exc_info=True)

app = Flask(__name__)
CORS(app)

# --- INICIALIZACIÓN DE REDIS ---
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    logging.critical("LA VARIABLE DE ENTORNO REDIS_URL NO ESTÁ CONFIGURADA. EL SERVICIO NO FUNCIONARÁ CORRECTAMENTE.")
    redis_client = None
else:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logging.info("Conexión con Redis establecida exitosamente.")
    except Exception as e:
        logging.critical(f"No se pudo conectar a Redis: {e}", exc_info=True)
        redis_client = None

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GCP_REGION)
    model_text = genai.GenerativeModel('gemini-1.5-flash')
    model_image = ImageGenerationModel.from_pretrained("imagegeneration@006")
    logging.info("Clientes de Google (Gemini, Storage, VertexAI) y ElevenLabs configurados.")
except Exception:
    logging.critical("ERROR FATAL AL CONFIGURAR CLIENTES DE GOOGLE.", exc_info=True)

REGLA_DE_VERACIDAD = "REGLA DE VERACIDAD CRÍTICA: Basa toda la información en hechos reales y bien documentados de fuentes confiables. Si el tema es una leyenda, un mito o una teoría, DEBES especificarlo claramente en el guion diciendo frases como 'según la leyenda...' o 'la teoría postula que...'. NO inventes eventos ni datos presentándolos como hechos."

PROMPTS_POR_NICHO = {
    "misterio_terror": f"**GANCHO INICIAL OBLIGATORIO:** La primera frase de la primera escena DEBE empezar obligatoriamente con '¿Sabías que...?'. A continuación, desarrolla una narración de suspenso y terror. {REGLA_DE_VERACIDAD}",
    "finanzas_emprendimiento": f"Redacta una narración inspiradora sobre una historia de éxito financiero, de emprendimiento o un tema financiero en tendencia. Utiliza un tono motivador y profesional. {REGLA_DE_VERACIDAD}",
    "tecnologia_ia": f"Genera una narración informativa y futurista sobre un avance en IA o tecnología, investigando en fuentes oficiales. El estilo debe ser didáctico y accesible. {REGLA_DE_VERACIDAD}",
    "documentales": f"Escribe una narración estrictamente objetiva, informativa y neutral sobre un tema social, cultural o histórico. El tono debe ser serio y documental, con un enfoque absoluto en hechos, fechas y análisis. {REGLA_DE_VERACIDAD}",
    "biblia_cristianismo": f"Redacta una narración inspiradora basada en pasajes bíblicos o reflexiones cristianas. Usa un tono respetuoso y espiritual. Basa la narración en interpretaciones teológicas aceptadas y no inventes hechos. {REGLA_DE_VERACIDAD}",
    "aliens_teorias": f"Crea una narración intrigante sobre una teoría conspirativa o un caso de contacto extraterrestre. Usa un estilo misterioso y especulativo. {REGLA_DE_VERACIDAD}",
    "tendencias_virales": f"Genera una narración dinámica sobre una tendencia viral actual. Explica el origen y contexto real de la tendencia. Usa un tono divertido y moderno. {REGLA_DE_VERACIDAD}",
    "politica": f"Redacta una narración seria y crítica sobre un tema político actual, usando un tono imparcial pero analítico. Cita hechos, estadísticas y consecuencias verificables. {REGLA_DE_VERACIDAD}",
    "anime_manga": f"Crea una narración apasionada sobre un anime, manga o su creador. Basa la información en la trama oficial de la obra o en datos biográficos confirmados. {REGLA_DE_VERACIDAD}"
}

def convertir_numeros_a_texto(texto):
    if not texto:
        return texto
    return re.sub(r'\d+', lambda match: num2words(int(match.group(0)), lang='es'), texto)

def upload_to_gcs(file_stream, destination_blob_name, content_type):
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        logging.info(f"Subiendo {len(file_stream)} bytes a GCS en {destination_blob_name}...")
        blob.upload_from_string(file_stream, content_type=content_type)
        logging.info(f"Subida completada. URL pública: {blob.public_url}")
        return blob.public_url
    except Exception as e:
        logging.error(f"¡FALLO CRÍTICO! No se pudo subir el archivo a GCS. Error: {e}", exc_info=True)
        raise

def safe_json_parse(raw_text):
    logging.info("Iniciando parseo de JSON robusto.")
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        text_to_parse = json_match.group(1)
        logging.info("Bloque JSON explícito ```json ... ``` encontrado.")
    else:
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1:
            text_to_parse = raw_text[start:end+1]
        else:
            logging.error(f"No se pudo encontrar un bloque JSON en la respuesta: {raw_text[:200]}")
            return None
    try:
        return json.loads(text_to_parse)
    except json.JSONDecodeError:
        logging.error(f"Fallo final de parseo JSON: {text_to_parse[:500]}", exc_info=True)
        return None

def _generate_and_upload_image(image_prompt, aspect_ratio):
    try:
        final_prompt = f"cinematic still, photorealistic, high detail of: {image_prompt}"
        images = model_image.generate_images(
            prompt=final_prompt, number_of_images=1, aspect_ratio=aspect_ratio,
            negative_prompt="text, watermark, logo, person, people, face, skin, deformed, blurry"
        )
        if not images:
            logging.warning(f"La API de imágenes no devolvió resultados para: {final_prompt}")
            return None
        return upload_to_gcs(images[0]._image_bytes, f"images/img_{uuid.uuid4()}.png", 'image/png')
    except Exception as e:
        logging.error(f"Excepción en _generate_and_upload_image: {e}", exc_info=True)
        return None

def _generate_audio_with_elevenlabs(text_input, voice_id):
    tts_url = f"{ELEVENLABS_API_URL}/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    data = {"text": text_input, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    
    response = requests.post(tts_url, json=data, headers=headers)

    if not response.ok:
        error_details = ""
        try:
            error_details = response.json()
        except json.JSONDecodeError:
            error_details = response.text
        logging.error(f"Error en la API de ElevenLabs. Código: {response.status_code}. Detalles: {error_details}")
        response.raise_for_status()

    content_type = response.headers.get('Content-Type', '')
    if 'audio/mpeg' not in content_type or not response.content:
        logging.error(
            f"La API de ElevenLabs no devolvió un archivo de audio válido. "
            f"Content-Type recibido: '{content_type}', "
            f"Tamaño del contenido: {len(response.content)} bytes."
        )
        raise ValueError("La respuesta de la API de voz estaba vacía o no era un archivo de audio válido.")

    logging.info(f"Audio recibido de ElevenLabs. Tamaño: {len(response.content)} bytes.")
    return upload_to_gcs(response.content, f"audio/audio_{uuid.uuid4()}.mp3", 'audio/mpeg')

def _generate_script_and_prepare_structure_task(job_id, prompt_final):
    def update_job_status(status, data=None, error=None):
        job_data = {"status": status}
        if data: job_data["result"] = data
        if error: job_data["error"] = error
        redis_client.set(job_id, json.dumps(job_data), ex=3600)

    try:
        logging.info(f"Trabajo de guion {job_id}: Iniciando generación en segundo plano.")
        update_job_status("processing")

        response = model_text.generate_content(prompt_final)
        parsed_json = safe_json_parse(response.text)

        if not (parsed_json and 'scenes' in parsed_json and isinstance(parsed_json['scenes'], list) and parsed_json['scenes']):
            logging.error(f"Trabajo {job_id}: La IA no generó un guion JSON válido. Respuesta: {response.text}")
            raise ValueError("La IA no pudo generar un guion válido. Intenta ajustar tu entrada.")

        scenes = parsed_json['scenes']
        logging.info(f"Trabajo {job_id}: Guion recibido de la IA. Estructurando {len(scenes)} escenas.")
        
        for scene in scenes:
            scene['id'] = str(uuid.uuid4())
            scene['imageUrl'] = None
            scene['videoUrl'] = None

        update_job_status("completed", data={"scenes": scenes})
        logging.info(f"Trabajo de guion {job_id}: Proceso completado y guardado en Redis.")

    except Exception as e:
        logging.error(f"Trabajo de guion {job_id} falló durante la generación en segundo plano: {e}", exc_info=True)
        update_job_status("error", error=str(e))

def _generate_audio_task(job_id, script, voice_id):
    def update_job_status(status, data=None, error=None):
        job_data = {"status": status}
        if data: job_data["result"] = data
        if error: job_data["error"] = error
        if redis_client:
            redis_client.set(job_id, json.dumps(job_data), ex=3600)

    try:
        logging.info(f"Trabajo de audio {job_id}: Iniciando generación en segundo plano.")
        update_job_status("processing")

        script_procesado = convertir_numeros_a_texto(script)
        logging.info(f"Trabajo {job_id}: Guion procesado para ElevenLabs: '{script_procesado[:80]}...'")
        
        audio_url = _generate_audio_with_elevenlabs(script_procesado, voice_id)
        if not audio_url:
            raise Exception("La función _generate_audio_with_elevenlabs no devolvió una URL.")

        update_job_status("completed", data={"audioUrl": audio_url})
        logging.info(f"Trabajo de audio {job_id}: Proceso completado. URL guardada en Redis.")

    except Exception as e:
        logging.error(f"Trabajo de audio {job_id} falló: {e}", exc_info=True)
        update_job_status("error", error=f"No se pudo generar el audio: {str(e)}")

@app.route("/")
def index():
    return "Backend de IA para Videos v19.1 'Corrección de Sintaxis' - Estable"

@app.route('/api/generate-initial-content', methods=['POST', 'OPTIONS'])
def generate_initial_content():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if not redis_client:
        return jsonify({"error": "El servicio de estado (Redis) no está disponible."}), 503
    
    try:
        data = request.get_json()
        if not data or not data.get('userInput', '').strip():
            return jsonify({"error": "El cuerpo de la solicitud es inválido o el campo de tema está vacío."}), 400

        output_format_instructions = """
        FORMATO DE SALIDA OBLIGATORIO:
        1. Tu respuesta DEBE SER ÚNICAMENTE un objeto JSON válido, sin texto adicional antes o después.
        2. El JSON debe tener una clave "scenes", que es una lista de objetos.
        3. Cada objeto de escena DEBE tener estas DOS claves:
           - "script": El texto del guion para esa escena en {idioma}.
           - "image_prompt": Una descripción CORTA y VISUAL en INGLÉS para un generador de imágenes (ej: "A dark mysterious forest at night with fog").
        REGLA CRÍTICA: Si dentro de "script" usas comillas dobles ("), DEBES escaparlas con una barra invertida (\\").
        """
        
        nicho = data.get('nicho')
        if not nicho or nicho not in PROMPTS_POR_NICHO:
            logging.error(f"Se recibió un nicho inválido o no existente: '{nicho}'")
            return jsonify({"error": f"El nicho '{nicho}' no es válido o no fue proporcionado."}), 400

        userInput = data.get('userInput')
        idioma = data.get('idioma', 'Español Latinoamericano')
        duracion_a_escenas = {"50": 4, "120": 6, "180": 8, "300": 10, "600": 15}
        numero_de_escenas = duracion_a_escenas.get(str(data.get('duracionVideo', '50')), 4)
        
        prompt_final = ""
        cta_instruction = "La última escena DEBE ser exclusivamente un llamado a la acción (CTA) claro y directo. Pide al espectador que 'nos siga en todas nuestras redes sociales para no perderse más contenido como este'."

        if data.get('tipoEntrada') == 'guion':
            prompt_template_guion = f"""
            ROL: Eres un editor de video y guionista experto.
            TAREA: Analiza el siguiente guion y reestructúralo en {numero_de_escenas} escenas en {idioma}.
            REGLAS: {cta_instruction}
            GUION: "{userInput}"
            {output_format_instructions.format(idioma=idioma)}
            """
            prompt_final = prompt_template_guion
        else:
            instruccion_base = PROMPTS_POR_NICHO[nicho]
            palabras_totales = int(data.get('duracionVideo', 50)) * 2.8
            palabras_por_escena = int(palabras_totales // numero_de_escenas)
            
            prompt_template_tema = f"""
            ROL: Eres un guionista experto para el nicho '{nicho}'.
            TAREA: Crea un guion sobre el tema principal: "{userInput}".
            IDIOMA: {idioma}.
            REGLAS OBLIGATORIAS:
            1. {instruccion_base}
            2. Genera EXACTAMENTE {numero_de_escenas} escenas. Cada una con aprox. {palabras_por_escena} palabras.
            3. CONSISTENCIA NARRATIVA CRÍTICA: Toda la historia debe centrarse en UN ÚNICO tema basado en "{userInput}". NO introduzcas temas no relacionados.
            4. {cta_instruction}
            
            {output_format_instructions.format(idioma=idioma)}
            """
            prompt_final = prompt_template_tema
        
        job_id = f"script_{uuid.uuid4()}" 
        initial_job_data = {"status": "pending", "jobId": job_id}
        redis_client.set(job_id, json.dumps(initial_job_data), ex=3600)
        
        thread = threading.Thread(target=_generate_script_and_prepare_structure_task, args=(job_id, prompt_final))
        thread.start()
        
        logging.info(f"Trabajo de guion {job_id} creado y enviado a segundo plano.")
        return jsonify({"jobId": job_id})

    except Exception as e:
        logging.error("Error en generate_initial_content.", exc_info=True)
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

@app.route('/api/job-status/<job_id>', methods=['GET'])
def get_content_job_status(job_id):
    if not redis_client:
        return jsonify({"error": "El servicio de estado (Redis) no está disponible."}), 503
        
    job_data_str = redis_client.get(job_id)
    if not job_data_str:
        return jsonify({"status": "not_found", "error": "Trabajo no encontrado o expirado"}), 404
    
    return jsonify(json.loads(job_data_str))

@app.route('/api/regenerate-scene-part', methods=['POST', 'OPTIONS'])
def regenerate_scene_part():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        data = request.get_json()
        scene = data.get('scene')
        part_to_regenerate = data.get('part')
        config = data.get('config')
        if not all([scene, part_to_regenerate, config]):
            return jsonify({"error": "Faltan datos en la solicitud"}), 400
        
        if part_to_regenerate == 'script':
            prompt = f"Reformula este texto de forma creativa y concisa, en Español Latinoamericano, manteniendo la idea central: '{scene.get('script')}'"
            response = model_text.generate_content(prompt)
            return jsonify({"newScript": response.text.strip().replace("`", "")})
        
        elif part_to_regenerate == 'media':
            image_prompt_from_scene = scene.get('image_prompt')
            if not image_prompt_from_scene:
                return jsonify({"error": "La escena no contiene un prompt de imagen para regenerar."}), 400
            
            aspect_ratio = '9:16' if config.get('resolucionVideo') == '9:16' else '16:9'
            logging.info(f"Regenerando imagen con prompt existente: '{image_prompt_from_scene}'")
            new_image_url = _generate_and_upload_image(image_prompt_from_scene, aspect_ratio)

            if not new_image_url:
                return jsonify({"error": "La IA no pudo generar una nueva imagen."}), 500
            return jsonify({"newImageUrl": new_image_url, "newVideoUrl": None})
            
    except Exception as e:
        logging.error(f"Error al regenerar parte de escena: {e}", exc_info=True)
        return jsonify({"error": f"Error interno al regenerar: {str(e)}"}), 500
    return jsonify({"error": "Parte no válida para regenerar."}), 400

@app.route('/api/generate-full-audio', methods=['POST', 'OPTIONS'])
def generate_full_audio():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if not redis_client:
        return jsonify({"error": "El servicio de estado (Redis) no está disponible."}), 503

    try:
        data = request.get_json()
        script = data.get('script')
        voice_id = data.get('voice', 'Wl3O9lmFSMgGFTTwuS6f')

        if not script or not script.strip():
            return jsonify({"error": "El guion es requerido"}), 400
        
        job_id = f"audio_{uuid.uuid4()}"
        initial_job_data = {"status": "pending", "jobId": job_id}
        redis_client.set(job_id, json.dumps(initial_job_data), ex=3600)
        
        thread = threading.Thread(target=_generate_audio_task, args=(job_id, script, voice_id))
        thread.start()
        
        logging.info(f"Trabajo de audio {job_id} creado.")
        return jsonify({"jobId": job_id})

    except Exception as e:
        logging.error(f"Error fatal en la generación de audio: {e}", exc_info=True)
        return jsonify({"error": f"Error interno del servidor durante la generación de audio: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
