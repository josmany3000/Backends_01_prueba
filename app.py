import os
import openai
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

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

        # --- INICIO: LÓGICA DE RESOLUCIÓN AÑADIDA ---
        # Determinar el tamaño de la imagen basado en la resolución enviada desde el frontend.
        resolucion = data.get('resolucion')
        if resolucion == '9:16':
            image_size = '1024x1792'  # Formato vertical para TikTok/Shorts/Reels
        elif resolucion == '16:9':
            image_size = '1792x1024'  # Formato horizontal para YouTube
        else:
            image_size = '1024x1024'  # Formato cuadrado por defecto
        # --- FIN: LÓGICA DE RESOLUCIÓN AÑADIDA ---

        # El prompt para GPT-4, ahora mucho más detallado
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
        
        # Procesar el texto para separar escenas y prompts
        scenes_data = parse_generated_text(generated_text)

        # Generar imágenes para cada escena
        for scene in scenes_data:
            if scene.get('image_prompt'):
                response_dalle = client.images.generate(
                    model="dall-e-3",
                    prompt=scene['image_prompt'],
                    n=1,
                    size=image_size, # <-- MODIFICADO: Usa la variable de tamaño dinámico
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
            if scene.get('script'):
                response_tts = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=scene['script']
                )
                # Guardar el archivo de audio
                audio_filename = f"{scene['id']}.mp3"
                audio_path = os.path.join(AUDIO_DIR, audio_filename)
                response_tts.stream_to_file(audio_path)
                # Devolver la URL desde donde se puede acceder al archivo
                scene['audioUrl'] = f"{request.host_url}audio/{audio_filename}"
        
        return jsonify({"scenes": scenes})

    except Exception as e:
        print(f"Error en generate_audio: {e}")
        return jsonify({"error": str(e)}), 500

# Ruta para servir los archivos de audio
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

# ... Aquí irían los endpoints para regenerar guion/imagen del Paso 2
# y para el SEO del Paso 6, siguiendo una lógica similar.

# --- FUNCIONES AUXILIARES ---
def definir_tono_por_nicho(nicho):
    # Mapea un nicho a una instrucción de tono para la IA
    tonos = {
        "misterio": "enigmático y que genere suspenso",
        "finanzas": "profesional, claro y confiable",
        "tecnologia": "innovador, futurista y fácil de entender",
        "documentales": "informativo, objetivo y narrativo",
        "anime": "entusiasta y conocedor, como un verdadero fan",
        # Puedes añadir más nichos del frontend aquí
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
    # Primero, separar los ganchos del resto
    parts = text.split("[SCENE_BREAK]")
    # La lógica aquí dividiría cada parte en guion y prompt de imagen,
    # y los añadiría a la lista `scenes_list`.
    # Por simplicidad, este es un ejemplo básico.
    for i, part in enumerate(parts):
        if "[PROMPT_IMAGEN]" in part:
            try:
                script, image_prompt = part.split("[PROMPT_IMAGEN]")
                scenes_list.append({
                    "id": f"scene_{uuid.uuid4()}",
                    "script": script.strip(),
                    "image_prompt": image_prompt.strip(),
                    "imageUrl": None,
                    "audioUrl": None
                })
            except ValueError:
                print(f"Advertencia: No se pudo procesar la parte: {part}")
                # Podrías decidir ignorar esta parte o manejarla de otra forma
                continue
    return scenes_list

if __name__ == '__main__':
    app.run(debug=True, port=5001)
