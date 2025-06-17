import os
import openai
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import json # NUEVO: Necesario para parsear respuestas de la IA

# --- CONFIGURACIÓN INICIAL ---
load_dotenv()
app = Flask(__name__)
# Habilita CORS para permitir que el frontend se comunique con este servidor
CORS(app) 
# Inicializa el cliente de OpenAI
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print("Error: No se pudo inicializar el cliente de OpenAI. Revisa tu clave de API en el archivo .env")
    print(e)


# Directorio para guardar archivos de audio generados
AUDIO_DIR = os.path.join(os.getcwd(), "audio_files")
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- RUTAS DE LA API ---

@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    """
    Endpoint para el Paso 1. 
    Recibe la configuración inicial y devuelve un guion y prompts de imagen.
    """
    try:
        data = request.json
        num_scenes = 7 # Por defecto
        if data.get('duracion') == '120': num_scenes = 14
        if data.get('duracion') == '180': num_scenes = 21

        # Determinar el tamaño de la imagen basado en la resolución enviada desde el frontend.
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
           - Al final del guion de la escena, añade una llamada a la acción o una frase de cierre (ej. "síguenos para más").
           - Después del guion, escribe en una nueva línea un prompt visualmente descriptivo para DALL-E 3 que represente la escena. Etiqueta este prompt con [PROMPT_IMAGEN].
        
        Usa "[SCENE_BREAK]" para separar cada escena completa (guion + prompt de imagen).
        Ejemplo de una escena:
        Guion de la escena aquí. ¡No te pierdas el próximo video!
        [PROMPT_IMAGEN] Un robot futurista escribiendo en una computadora holográfica, estilo cinematográfico.
        """

        response_gpt = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        generated_text = response_gpt.choices[0].message.content
        
        scenes_data = parse_generated_text(generated_text)

        for scene in scenes_data:
            if scene.get('image_prompt'):
                response_dalle = client.images.generate(
                    model="dall-e-3",
                    prompt=scene['image_prompt'],
                    n=1,
                    size=image_size,
                    quality="standard"
                )
                scene['imageUrl'] = response_dalle.data[0].url
        
        return jsonify({"scenes": scenes_data})

    except Exception as e:
        print(f"Error en generate_initial_content: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """
    Endpoint para el Paso 3.
    Recibe un conjunto de escenas y una voz, y genera el audio para cada una.
    """
    try:
        data = request.json
        scenes = data.get('scenes', [])
        voice = data.get('voice', 'nova')

        for scene in scenes:
            if scene.get('script') and not scene.get('audioUrl'): # Optimización: solo genera si no existe
                response_tts = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=scene['script']
                )
                audio_filename = f"{scene['id']}.mp3"
                audio_path = os.path.join(AUDIO_DIR, audio_filename)
                response_tts.stream_to_file(audio_path)
                scene['audioUrl'] = f"{request.host_url}audio/{audio_filename}"
        
        return jsonify({"scenes": scenes})

    except Exception as e:
        print(f"Error en generate_audio: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================================
# ===    NUEVO ENDPOINT FUNCIONAL PARA REGENERAR GUION/IMAGEN/AUDIO     ===
# =========================================================================
@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
    """
    Endpoint para el Paso 2 y 3 (regeneración).
    Recibe el ID de una escena, la parte a regenerar ('script', 'image', 'audio')
    y los datos necesarios, y devuelve la parte actualizada.
    """
    try:
        data = request.json
        part_to_regenerate = data.get('part') # 'script', 'image', o 'audio'
        scene = data.get('scene')             # El objeto de la escena actual
        config = data.get('config')           # La configuración inicial

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

            # El prompt de la imagen se basa en el guion ACTUAL de la escena
            image_prompt = f"Crea una imagen visualmente impactante para un video sobre '{config.get('tema')}'. La imagen debe ilustrar esta idea: '{scene.get('script')}'. Estilo: {definir_tono_por_nicho(config.get('nicho'))}, cinematográfico, alta calidad."
            
            response_dalle = client.images.generate(
                model="dall-e-3", prompt=image_prompt, n=1, size=image_size, quality="standard"
            )
            new_image_url = response_dalle.data[0].url
            return jsonify({"newImageUrl": new_image_url})

        # --- REGENERAR AUDIO ---
        elif part_to_regenerate == 'audio':
            voice = config.get('voice', 'nova')
            response_tts = client.audio.speech.create(
                model="tts-1", voice=voice, input=scene.get('script')
            )
            audio_filename = f"{scene['id']}.mp3"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            response_tts.stream_to_file(audio_path)
            new_audio_url = f"{request.host_url}audio/{audio_filename}"
            return jsonify({"newAudioUrl": new_audio_url})

        else:
            return jsonify({"error": "Parte a regenerar no válida"}), 400

    except Exception as e:
        print(f"Error en regenerate_scene_part: {e}")
        return jsonify({"error": str(e)}), 500
        
# =========================================================================
# ===          NUEVO ENDPOINT FUNCIONAL PARA GENERAR SEO                ===
# =========================================================================
@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
    """
    Endpoint para el Paso 6.
    Recibe el guion completo del video y el nicho, y genera Título, Descripción y Hashtags.
    """
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
        2. "descripcion": Una descripción optimizada para el algoritmo. Debe incluir un resumen atractivo, una llamada a la acción (suscribirse, seguir, comentar) y usar palabras clave relevantes del guion.
        3. "hashtags": Una lista de 10-15 hashtags relevantes y populares, mezclando hashtags generales y específicos del nicho. Devuélvelos como un único string separados por espacios (ej: "#finanzas #inversion #bitcoin").

        RESPUESTA JSON:
        """
        response_gpt = client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"}, # NUEVO: Forzar respuesta JSON
            messages=[{"role": "system", "content": prompt}]
        )

        # La respuesta ya viene en formato JSON, solo necesitamos parsearla
        seo_data = json.loads(response_gpt.choices[0].message.content)

        return jsonify(seo_data)

    except Exception as e:
        print(f"Error en generate_seo: {e}")
        return jsonify({"error": str(e)}), 500


# Ruta para servir los archivos de audio
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# --- FUNCIONES AUXILIARES ---
def definir_tono_por_nicho(nicho):
    # Mapea un nicho a una instrucción de tono para la IA
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
    # Esta función es crucial. Debe procesar el texto plano de GPT
    # y convertirlo en una estructura de datos limpia.
    scenes_list = []
    clean_text = text.replace("[GANCHO]", "").strip()
    parts = clean_text.split("[SCENE_BREAK]")
    
    for i, part in enumerate(parts):
        if "[PROMPT_IMAGEN]" in part:
            try:
                # Usar rsplit por si el guion contiene la etiqueta por error
                script, image_prompt = part.rsplit("[PROMPT_IMAGEN]", 1)
                scenes_list.append({
                    "id": f"scene_{uuid.uuid4()}",
                    "script": script.strip(),
                    "image_prompt": image_prompt.strip(), # Guardamos el prompt original
                    "imageUrl": None,
                    "audioUrl": None
                })
            except ValueError:
                print(f"Advertencia: No se pudo procesar la parte: {part}")
                continue
    return scenes_list

if __name__ == '__main__':
    app.run(debug=True, port=5001)

