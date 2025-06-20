import os
import openai
import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json
from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO

# --- CONFIGURACIÓN INICIAL ---
load_dotenv()
app = Flask(__name__)
CORS(app) 

# Inicializa el cliente de OpenAI
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print("Error inicializando OpenAI:", e)

# Configura Google Cloud Storage
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_CREDENTIALS = json.loads(os.getenv("GCS_CREDENTIALS_JSON")) if os.getenv("GCS_CREDENTIALS_JSON") else None

# Función para subir archivos a GCS
def upload_to_gcs(data, blob_name, content_type):
    """Sube datos a Google Cloud Storage y devuelve URL pública"""
    try:
        credentials = service_account.Credentials.from_service_account_info(GCS_CREDENTIALS)
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_string(
            data,
            content_type=content_type
        )
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Error subiendo a GCS: {e}")
        raise

# --- RUTAS DE LA API ---

@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    try:
        data = request.json
        num_scenes = 7
        if data.get('duracion') == '120': num_scenes = 14
        if data.get('duracion') == '180': num_scenes = 21

        resolucion = data.get('resolucion')
        if resolucion == '9:16':
            image_size = '1024x1792'
        elif resolucion == '16:9':
            image_size = '1792x1024'
        else:
            image_size = '1024x1024'

        prompt = f"""
        Eres un guionista experto para videos virales de redes sociales en el nicho de '{data.get('nicho')}'.
        Tu tono debe ser {definir_tono_por_nicho(data.get('nicho'))}.
        El video debe ser sobre el tema: '{data.get('tema')}'.
        El idioma del guion debe ser: {data.get('idioma')}.

        TAREA:
        1. Escribe {data.get('cantidadGanchos')} ganchos virales de una sola frase. Etiqueta cada uno con [GANCHO].
        2. Escribe un guion para un video de {num_scenes} escenas. Para cada escena:
           - Escribe un guion corto y conciso.
           - Al final del guion de la escena, añade una llamada a la acción o una frase de cierre.
           - Después del guion, escribe un prompt visualmente descriptivo para DALL-E 3. Etiqueta con [PROMPT_IMAGEN].
        
        Usa "[SCENE_BREAK]" para separar cada escena completa.
        """

        response_gpt = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        generated_text = response_gpt.choices[0].message.content
        
        scenes_data = parse_generated_text(generated_text)

        for scene in scenes_data:
            if scene.get('image_prompt'):
                # Generar imagen con DALL-E
                response_dalle = client.images.generate(
                    model="dall-e-3",
                    prompt=scene['image_prompt'],
                    n=1,
                    size=image_size,
                    quality="standard"
                )
                
                # Descargar imagen y subir a GCS
                image_url = response_dalle.data[0].url
                image_response = requests.get(image_url)
                
                if image_response.status_code == 200:
                    public_url = upload_to_gcs(
                        image_response.content,
                        f"images/{scene['id']}.png",
                        "image/png"
                    )
                    scene['imageUrl'] = public_url
        
        return jsonify({"scenes": scenes_data})

    except Exception as e:
        print(f"Error en generate_initial_content: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    try:
        data = request.json
        scenes = data.get('scenes', [])
        voice = data.get('voice', 'nova')

        for scene in scenes:
            if scene.get('script') and not scene.get('audioUrl'):
                # Generar audio en memoria
                response_tts = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=scene['script']
                )
                
                audio_buffer = BytesIO()
                response_tts.stream_to_file(audio_buffer)
                audio_buffer.seek(0)
                
                # Subir a GCS
                audio_url = upload_to_gcs(
                    audio_buffer.getvalue(),
                    f"audios/{scene['id']}.mp3",
                    "audio/mpeg"
                )
                scene['audioUrl'] = audio_url
        
        return jsonify({"scenes": scenes})

    except Exception as e:
        print(f"Error en generate_audio: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================================
# ===    ENDPOINT PARA REGENERAR GUION/IMAGEN/AUDIO     ===
# =========================================================================
@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
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
            response_gpt = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            new_script = response_gpt.choices[0].message.content.strip()
            return jsonify({"newScript": new_script})

        # --- REGENERAR IMAGEN ---
        elif part_to_regenerate == 'image':
            resolucion = config.get('resolucion')
            if resolucion == '9:16': image_size = '1024x1792'
            elif resolucion == '16:9': image_size = '1792x1024'
            else: image_size = '1024x1024'

            image_prompt = f"Crea una imagen visualmente impactante para un video sobre '{config.get('tema')}'. La imagen debe ilustrar esta idea: '{scene.get('script')}'. Estilo: {definir_tono_por_nicho(config.get('nicho'))}, cinematográfico, alta calidad."
            
            response_dalle = client.images.generate(
                model="dall-e-3", prompt=image_prompt, n=1, size=image_size, quality="standard"
            )
            
            # Descargar y subir a GCS
            image_url = response_dalle.data[0].url
            image_response = requests.get(image_url)
            
            if image_response.status_code == 200:
                public_url = upload_to_gcs(
                    image_response.content,
                    f"images/{scene['id']}.png",
                    "image/png"
                )
                return jsonify({"newImageUrl": public_url})

        # --- REGENERAR AUDIO ---
        elif part_to_regenerate == 'audio':
            voice = config.get('voice', 'nova')
            response_tts = client.audio.speech.create(
                model="tts-1", voice=voice, input=scene.get('script')
            )
            
            audio_buffer = BytesIO()
            response_tts.stream_to_file(audio_buffer)
            audio_buffer.seek(0)
            
            # Subir a GCS
            audio_url = upload_to_gcs(
                audio_buffer.getvalue(),
                f"audios/{scene['id']}.mp3",
                "audio/mpeg"
            )
            return jsonify({"newAudioUrl": audio_url})

        else:
            return jsonify({"error": "Parte a regenerar no válida"}), 400

    except Exception as e:
        print(f"Error en regenerate_scene_part: {e}")
        return jsonify({"error": str(e)}), 500
        
# =========================================================================
# ===          ENDPOINT PARA GENERAR SEO                ===
# =========================================================================
@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
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
        Proporciona los siguientes tres elementos en un formato JSON válido:
        1. "titulo": Un título corto, viral y que genere curiosidad (máximo 70 caracteres).
        2. "descripcion": Una descripción optimizada para el algoritmo.
        3. "hashtags": Una lista de 10-15 hashtags relevantes y populares.
        """
        response_gpt = client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}]
        )

        seo_data = json.loads(response_gpt.choices[0].message.content)
        return jsonify(seo_data)

    except Exception as e:
        print(f"Error en generate_seo: {e}")
        return jsonify({"error": str(e)}), 500

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

def parse_generated_text(text):
    scenes_list = []
    clean_text = text.replace("[GANCHO]", "").strip()
    parts = clean_text.split("[SCENE_BREAK]")
    
    for i, part in enumerate(parts):
        if "[PROMPT_IMAGEN]" in part:
            try:
                script, image_prompt = part.rsplit("[PROMPT_IMAGEN]", 1)
                scenes_list.append({
                    "id": f"scene_{uuid.uuid4()}",
                    "script": script.strip(),
                    "image_prompt": image_prompt.strip(),
                    "imageUrl": None,
                    "audioUrl": None
                })
            except ValueError:
                continue
    return scenes_list

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)))