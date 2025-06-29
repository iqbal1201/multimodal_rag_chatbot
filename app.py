import uuid, os
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, CancellationReason
import io
from pydub import AudioSegment


# Flask app init
app = Flask(__name__)
CORS(app)

# ENV config
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT")
API_VERSION = os.getenv("API_VERSION", "2025-01-01-preview")
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

# Azure OpenAI Client via Entra ID
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION,
)

# Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
You are AI Assistant to provide answer to user question regarding company policy PT Rekso Nasional Food
2023 - 2025.
Use the information provided in the context to answer the user's question.
If you don't know the answer, say you don't know. Do not make up answers.
Don't provide anything outside the given context.
Respond in {language}

Context:
{context}

Question:
{question}

Answer directly. No small talk.
"""

# Vectorstore + Embeddings (FAISS)
DB_FAISS_PATH = "vectorstore/rekso_2023_2025_v2"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    user_language = data.get('language', 'en-US')  # default to English

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Map language code to natural language
        lang_instruction = {
            'en-US': 'English',
            'id-ID': 'Bahasa Indonesia',
            'zh-CN': 'Chinese',
        }.get(user_language, 'English')

        # Retrieve documents
        retrieved_docs = retriever.invoke(user_message)
        if not retrieved_docs:
            print("No relevant documents found.")
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"\n=== Doc {i+1} ===")
                print(doc.page_content)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        # # Build prompt
        # filled_prompt = CUSTOM_PROMPT_TEMPLATE.format(
        #     context=context_text,
        #     question=user_message,
        #     language=lang_instruction
        # )

        # # Query Azure OpenAI
        # response = client.chat.completions.create(
        #     model=DEPLOYMENT_NAME,
        #     messages=[{"role": "user", "content": filled_prompt}],
        #     temperature=0.5,
        #     max_tokens=800
        # )

        # # Prepare messages
        # messages = [
        #     {"role": "system", "content": f"You are an AI assistant for company policy PT Rekso Nasional Food 2023-2025. Use the provided context to answer the user's question. If the answer is not in the context, respond with 'I don't know.' using {lang_instruction}. Do not make up answers. Respond in {lang_instruction}. Answer directly without small talk."},
        #     {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{user_message}"}
        # ]

        # Build system prompt with embedded context
        system_prompt = f"""
                You are an AI assistant for company policy PT Rekso Nasional Food 2023-2025.
                Use {lang_instruction} to give response.
                Use the following context to answer the user's question.
                If the answer is not in the context, respond with "I don't know."
                Respond in {lang_instruction}. Answer directly without small talk.

                Context:
                {context_text}
                """

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Query Azure OpenAI
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.0,  # better for factual policy QA
            max_tokens=800
        )

        answer_text = response.choices[0].message.content

        return jsonify({"response": answer_text})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
    
# STT Endpoint 
@app.route('/stt', methods=['POST'])
def stt():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    # Use .wav for Azure Speech recognition, as it's typically preferred or required
    # for certain configurations. The frontend is sending webm, so conversion might be needed,
    # but for simplicity, let's try saving as wav and assuming browser encoding is compatible or Azure handles it.
    # If issues persist, consider explicit client-side encoding to WAV or server-side conversion.
    filename = f"uploads/{uuid.uuid4()}.wav" 
    audio_file.save(filename)

    try:
        speech_config = SpeechConfig(subscription=speech_key, region=speech_region)
        # You can try to set the speech recognition language if you pass it from the frontend
        # For example: speech_config.speech_recognition_language = request.headers.get('X-Speech-Language', 'en-US') 
        # Or from the form data if you append it: request.form.get('language', 'en-US')
        
        audio_input = AudioConfig(filename=filename)
        speech_recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

        # Use recognize_once_async().get() for synchronous waiting (blocking)
        # In a production setting, you might want to use async Flask + threading for long operations
        result = speech_recognizer.recognize_once_async().get() 

        if result.reason == ResultReason.RecognizedSpeech:
            return jsonify({"text": result.text})
        elif result.reason == ResultReason.NoMatch:
            return jsonify({"error": "Speech could not be recognized. No match."}), 400
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech Recognition Canceled: Reason={cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                print(f"Cancellation ErrorDetails: {cancellation_details.error_details}")
                # You might want to log cancellation_details.error_details for debugging
            return jsonify({"error": f"Speech recognition canceled: {cancellation_details.reason}"}), 400
        else:
            return jsonify({"error": "Unknown speech recognition error."}), 400
    except Exception as e:
        print(f"Error during STT: {e}")
        return jsonify({"error": f"An error occurred during speech-to-text processing: {str(e)}"}), 500
    finally:
        # Clean up the saved audio file
        if os.path.exists(filename):
            os.remove(filename)
    
## TTS Endpoint
# TTS Endpoint (FIXED)
@app.route('/tts', methods=['POST'])
def tts():
    text = request.json.get('text')
    lang = request.json.get('language', 'en-US') 

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        speech_config = SpeechConfig(subscription=speech_key, region=speech_region)
        
        speech_config.speech_synthesis_voice_name = {
            'en-US': 'en-US-JennyNeural',
            'id-ID': 'id-ID-GadisNeural',
            'zh-CN': 'zh-CN-XiaoxiaoNeural'
        }.get(lang, 'en-US-JennyNeural')

        # Create a synthesizer without an AudioConfig specified for a file/stream
        # The synthesized audio will be available in the result object itself.
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None) 

        # Perform the synthesis. The 'get()' method of the result object
        # will now contain the audio data if successful.
        result = synthesizer.speak_text_async(text).get()

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            # Get the audio data directly from the result object
            audio_data = result.audio_data
            
            # Return the raw audio data as a Flask response
            return Response(audio_data, mimetype='audio/mpeg', status=200)
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: Reason={cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                print(f"Cancellation ErrorDetails: {cancellation_details.error_details}")
            return jsonify({"error": f"Speech synthesis canceled: {cancellation_details.reason}"}), 400
        else:
            print(f"TTS: Unknown reason for cancellation: {result.reason}")
            return jsonify({"error": "Unknown speech synthesis error."}), 400

    except Exception as e:
        print(f"Error during TTS: {e}")
        return jsonify({"error": f"An error occurred during text-to-speech processing: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
