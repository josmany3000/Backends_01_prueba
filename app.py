# app.py (Versión Mejorada y Optimizada)

import os
import openai
import uuid
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURACIÓN INICIAL Y CLIENTES GLOBALES ---
load_dotenv()
app = Flask(__name__)
# Se recomienda ser más específico en producción, ej: CORS(app, origins=["https://tu-dominio.com"])
CORS(app) 

# Inicializa los clientes UNA SOLA VEZ para reutilizarlos en toda la aplicación.
# Esto es mucho más eficiente que crearlos en cada petición.
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Cliente de OpenAI inicializado correctamente.")
except Exception as e:
    openai_client = None
    print(f"FATAL: Error al inicializar el cliente de OpenAI: {e}")

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_CREDENTIALS_JSON = os.getenv("GCS_CREDENTIALS_JSON")

gcs_bucket = None
if GCS_CREDENTIALS_JSON:
    try:
        gcs_credentials_info = json.loads(GCS_CREDENTIALS_JSON)
        gcs_credentials = service_account.Credentials.from_service_account_info(gcs_credentials_info)
        gcs_client = storage.Client(credentials=gcs_credentials)
        gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        print(f"Cliente de Google Cloud Storage inicializado y conectado al bucket '{GCS_BUCKET_NAME}'.")
    except Exception as e:
        print(f"FATAL: Error al inicializar el cliente de GCS: {e}")
else:
    print("FATAL: No se encontraron las credenciales GCS_CREDENTIALS_JSON en las variables de entorno.")


# --- FUNCIONES AUXILIARES ---

def definir_tono_por_nicho(nicho):
    tonos = {
        "misterio": "enigmático y que genere suspenso",
        "finanzas": "profesional, claro y confiable",
        "tecnologia": "innovador, futurista y fácil de entender",
        "documentales": "informativo, objetivo y narrativo",
        "anime": "entusiasta y conocedor, como un verdadero fan",
        "biblia": "respetuoso, solemne e inspirador",
        "extraterrestres": "misterioso, especulativo y abierto a teorías",
        "tendencias": "moderno, enérgico y llamativo",
        "politica": "objetivo, analítico y equilibrado",
    }
    return tonos.get(nicho, "neutral y atractivo")

def upload_to_gcs(data, blob_name, content_type):
    """Sube datos a Google Cloud Storage usando el bucket global."""
    if not gcs_bucket:
        raise Exception("El bucket de Google Cloud Storage no está inicializado.")
    try:
        blob = gcs_bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Error subiendo a GCS (blob: {blob_name}): {e}")
        raise

def process_scene_image(scene, image_size):
    """Función para generar imagen y subirla a GCS para UNA SOLA escena."""
    try:
        response_dalle = openai_client.images.generate(
            model="dall-e-3",
            prompt=scene['image_prompt'],
            n=1,
            size=image_size,
            quality="standard"
        )
        image_url = response_dalle.data[0].url
        image_response = requests.get(image_url, timeout=60)
        image_response.raise_for_status()
        
        public_url = upload_to_gcs(
            image_response.content,
            f"images/{scene['id']}.png",
            "image/png"
        )
        scene['imageUrl'] = public_url
    except Exception as e:
        print(f"Error procesando imagen para la escena {scene['id']}: {e}")
        scene['imageUrl'] = None  # Marcar la imagen como fallida
    return scene

def process_scene_audio(scene, voice):
    """Función para generar audio y subirlo a GCS para UNA SOLA escena."""
    try:
        response_tts = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=scene['script']
        )
        
        # Trabajar con el audio en memoria
        audio_buffer = BytesIO()
        response_tts.stream_to_file(audio_buffer)
        audio_buffer.seek(0)
        
        audio_url = upload_to_gcs(
            audio_buffer.getvalue(),
            f"audios/{scene['id']}.mp3",
            "audio/mpeg"
        )
        scene['audioUrl'] = audio_url
    except Exception as e:
        print(f"Error procesando audio para la escena {scene['id']}: {e}")
        scene['audioUrl'] = None # Marcar el audio como fallido
    return scene


# --- RUTAS DE LA API ---

@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    if not openai_client:
        return jsonify({"error": "El servicio de IA no está disponible."}), 503

    try:
        data = request.json
        num_scenes = 7
        if data.get('duracion') == '120': num_scenes = 14
        if data.get('duracion') == '180': num_scenes = 21

        resolucion = data.get('resolucion', '1:1')
        image_size_map = {'9:16': '1024x1792', '16:9': '1792x1024'}
        image_size = image_size_map.get(resolucion, '1024x1024')

        prompt = f"""
        Eres un guionista experto para videos virales de redes sociales en el nicho de '{data.get('nicho')}'.
        Tu tono debe ser {definir_tono_por_nicho(data.get('nicho'))}.
        El video debe ser sobre el tema: '{data.get('tema')}'.
        El idioma del guion debe ser: {data.get('idioma')}.

        TAREA:
        1. Escribe {data.get('cantidadGanchos')} ganchos virales de una sola frase.
        2. Escribe un guion para un video de {num_scenes} escenas.
        
        FORMATO DE SALIDA OBLIGATORIO:
        Devuelve un único objeto JSON válido. El objeto debe tener una clave "scenes", que es una lista de objetos.
        Cada objeto de escena en la lista debe tener exactamente dos claves:
        - "script": El guion corto y conciso para la escena.
        - "image_prompt": Un prompt visualmente descriptivo y detallado para DALL-E 3 que ilustre el guion.
        
        Ejemplo de la estructura de un objeto de escena:
        {{"script": "Este es el guion de la escena 1.", "image_prompt": "Un astronauta flotando en el espacio, estilo fotorrealista."}}

        Asegúrate de que la salida sea solo el objeto JSON, sin texto adicional antes o después.
        """
        response_gpt = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Carga el JSON directamente, mucho más robusto que parsear texto
        content = json.loads(response_gpt.choices[0].message.content)
        scenes_data = content.get('scenes', [])

        # Asignar IDs únicos a cada escena
        for scene in scenes_data:
            scene['id'] = f"scene_{uuid.uuid4()}"
            scene['imageUrl'] = None
            scene['audioUrl'] = None

        # --- PROCESAMIENTO EN PARALELO ---
        # Usamos ThreadPoolExecutor para procesar todas las imágenes al mismo tiempo.
        # Esto es crucial para no exceder los timeouts en plataformas como Render.
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Enviamos cada escena a procesar en un hilo separado
            futures = [executor.submit(process_scene_image, scene, image_size) for scene in scenes_data]
            # Recolectamos los resultados
            processed_scenes = [future.result() for future in futures]
        
        return jsonify({"scenes": processed_scenes})

    except Exception as e:
        print(f"Error en generate_initial-content: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    if not openai_client:
        return jsonify({"error": "El servicio de IA no está disponible."}), 503
        
    try:
        data = request.json
        scenes = data.get('scenes', [])
        voice = data.get('voice', 'nova')

        # Filtramos las escenas que realmente necesitan audio
        scenes_to_process = [s for s in scenes if s.get('script') and not s.get('audioUrl')]

        # --- PROCESAMIENTO EN PARALELO ---
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_scene_audio, scene, voice): scene for scene in scenes_to_process}
            
            processed_results = {}
            for future in futures:
                # Obtenemos la escena procesada
                processed_scene = future.result()
                processed_results[processed_scene['id']] = processed_scene

        # Actualizamos la lista original de escenas con los resultados procesados
        for i, scene in enumerate(scenes):
            if scene['id'] in processed_results:
                scenes[i] = processed_results[scene['id']]

        return jsonify({"scenes": scenes})

    except Exception as e:
        print(f"Error en generate_audio: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
    if not openai_client:
        return jsonify({"error": "El servicio de IA no está disponible."}), 503
        
    try:
        data = request.json
        part_to_regenerate = data.get('part')
        scene = data.get('scene')
        config = data.get('config')

        if not all([part_to_regenerate, scene, config]):
            return jsonify({"error": "Faltan datos en la petición"}), 400

        # --- REGENERAR GUION ---
        if part_to_regenerate == 'script':
            prompt = f"""
            Eres un guionista experto. Reescribe el siguiente guion para una escena de un video sobre '{config.get('tema')}' en el nicho '{config.get('nicho')}'.
            El tono debe ser {definir_tono_por_nicho(config.get('nicho'))} y en el idioma {config.get('idioma')}.
            Hazlo conciso y potente.
            
            GUION ANTIGUO: "{scene.get('script')}"
            
            NUEVO GUION:
            """
            response_gpt = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            new_script = response_gpt.choices[0].message.content.strip()
            return jsonify({"newScript": new_script})

        # --- REGENERAR IMAGEN (usa la función auxiliar de una sola escena) ---
        elif part_to_regenerate == 'image':
            resolucion = config.get('resolucion', '1:1')
            image_size_map = {'9:16': '1024x1792', '16:9': '1792x1024'}
            image_size = image_size_map.get(resolucion, '1024x1024')
            
            # Para la regeneración, necesitamos un prompt de imagen. Si no lo tenemos, lo creamos.
            image_prompt = scene.get('image_prompt')
            if not image_prompt:
                 image_prompt = f"Crea una imagen visualmente impactante para un video sobre '{config.get('tema')}'. La imagen debe ilustrar esta idea: '{scene.get('script')}'. Estilo: {definir_tono_por_nicho(config.get('nicho'))}, cinematográfico, alta calidad."
            scene['image_prompt'] = image_prompt

            updated_scene = process_scene_image(scene, image_size)
            return jsonify({"newImageUrl": updated_scene.get('imageUrl')})

        # --- REGENERAR AUDIO (usa la función auxiliar de una sola escena) ---
        elif part_to_regenerate == 'audio':
            voice = config.get('voice', 'nova')
            updated_scene = process_scene_audio(scene, voice)
            return jsonify({"newAudioUrl": updated_scene.get('audioUrl')})

        else:
            return jsonify({"error": "Parte a regenerar no válida"}), 400

    except Exception as e:
        print(f"Error en regenerate_scene_part: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
    if not openai_client:
        return jsonify({"error": "El servicio de IA no está disponible."}), 503
        
    try:
        data = request.json
        guion = data.get('guion')
        nicho = data.get('nicho')

        prompt = f"""
        Eres un experto en marketing digital y SEO para redes sociales como YouTube, TikTok e Instagram, especializado en el nicho de '{nicho}'.
        Basado en el siguiente guion de un video, genera el contenido para su publicación.

        GUION:
        ---
        {guion}
        ---

        TAREA:
        Proporciona los siguientes tres elementos en un formato JSON válido con las claves "titulo", "descripcion" y "hashtags":
        1. "titulo": Un título corto, viral y que genere curiosidad (máximo 70 caracteres).
        2. "descripcion": Una descripción optimizada para el algoritmo. Incluye un Call to Action.
        3. "hashtags": Una cadena de texto con 10-15 hashtags relevantes y populares, separados por espacios.
        """
        response_gpt = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}]
        )

        seo_data = json.loads(response_gpt.choices[0].message.content)
        return jsonify(seo_data)

    except Exception as e:
        print(f"Error en generate_seo: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Render establece la variable de entorno PORT. Para desarrollo local, usamos 5001.
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
