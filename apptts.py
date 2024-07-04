from flask import Flask, jsonify, request
from threading import Thread
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone
import spacy
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from playsound import playsound

app = Flask(__name__)

# Initialize Deepgram client
deepgram = DeepgramClient()
dg_connection = deepgram.listen.live.v("1")

# Load the intent classification model and tokenizer
intent_tokenizer = AutoTokenizer.from_pretrained("Oridak771/distilroberta-finetuned")
intent_model = AutoModelForSequenceClassification.from_pretrained("Oridak771/distilroberta-finetuned")

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Create intent classification pipeline
intent_classifier = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)

# Load and fit LabelEncoder on the original dataset intents
df = pd.read_csv('data.csv')
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['intent'])

# Dictionary to map encoded labels to intent names
label_map = {index: label for index, label in enumerate(label_encoder.classes_)}

# Variable to store the complete sentence
complete_sentence = ""
is_recording = False  # Variable to track recording state

INTENT_TO_API = {
    "check_balance": "http://localhost:5001/balance",
}

def on_open(self, open, **kwargs):
    print(f"\n\n{open}\n\n")

def on_message(self, result, **kwargs):
    global complete_sentence
    interim_sentence = result.channel.alternatives[0].transcript
    if len(interim_sentence) == 0:
        return

    print(f"speaker: {interim_sentence}")

    # Update the complete sentence with the latest interim result
    complete_sentence = interim_sentence

def on_metadata(self, metadata, **kwargs):
    print(f"\n\n{metadata}\n\n")

def on_speech_started(self, speech_started, **kwargs):
    print(f"\n\n{speech_started}\n\n")

def on_utterance_end(self, utterance_end, **kwargs):
    global complete_sentence
    print(f"\n\n{utterance_end}\n\n")

    if complete_sentence:
        # Get intent predictions
        intent_results = intent_classifier(complete_sentence)
        if intent_results:
            intent_result = intent_results[0]
            encoded_label = int(intent_result['label'].split('_')[-1])
            intent = label_map[encoded_label]
            score = intent_result['score']

            print(f"Text: {complete_sentence}")
            print(f"Label: {intent}, Score: {score:.4f}")

            # Get named entities using spaCy
            doc = nlp(complete_sentence)
            entities = [{"entity": ent.label_, "text": ent.text} for ent in doc.ents]

            print("Entities: ", entities)

            # Make API call based on the intent
            if intent in INTENT_TO_API:
                response = requests.get(INTENT_TO_API[intent])
                if response.status_code == 200:
                    api_response = response.json()
                    print(f"API Response: {api_response}")
                    # Correctly access the 'account_balance' from the API response
                    text_to_speak = api_response.get('account_balance', 'I could not get the balance.')
                    speak_text(text_to_speak)
                else:
                    print(f"API Error: {response.status_code}, {response.text}")
                    speak_text("There was an error retrieving the balance.")
            
        # Clear the complete sentence after processing
        complete_sentence = ""

def on_error(self, error, **kwargs):
    print(f"\n\n{error}\n\n")

def on_close(self, close, **kwargs):
    print(f"\n\n{close}\n\n")

dg_connection.on(LiveTranscriptionEvents.Open, on_open)
dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
dg_connection.on(LiveTranscriptionEvents.Error, on_error)
dg_connection.on(LiveTranscriptionEvents.Close, on_close)

options = LiveOptions(
    model="nova-2",
    punctuate=True,
    language="en-US",
    encoding="linear16",
    channels=1,
    sample_rate=16000,
    interim_results=True,
    utterance_end_ms="4500",  # Increased to 3 seconds
    vad_events=True,
)

def start_recording():
    global microphone
    dg_connection.start(options)
    microphone = Microphone(dg_connection.send)
    microphone.start()

def stop_recording():
    microphone.finish()
    dg_connection.finish()

@app.route('/toggle', methods=['POST'])
def toggle_recording():
    global is_recording
    if not is_recording:
        thread = Thread(target=start_recording)
        thread.start()
        is_recording = True
        return jsonify({"status": "recording started"})
    else:
        stop_recording()
        is_recording = False
        return jsonify({"status": "recording stopped"})

@app.route('/balance', methods=['GET'])
def get_balance():
    response = {
        'account_balance': '$500'
    }
    return jsonify(response)

def speak_text(text):
    tts_url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"  # Ensure this is the correct TTS endpoint
    api_key = "237332f30e5ef1eb5bcb6ce247943209c2451bbe"  # Ensure this is your valid API key

    headers = {
        'Authorization': f'Token {api_key}',
        'Content-Type': 'application/json',
    }
    
    data = {
        "text": text,  # Ensure this is the only key when sending text  # Adjust parameters as needed
    }

    response = requests.post(tts_url, headers=headers, json=data)
    if response.status_code == 200:
        audio_data = response.content
        play_audio(audio_data)
    else:
        print(f"TTS API Error: {response.status_code}, {response.text}")

def play_audio(audio_data):
    # Save the audio data to a temporary file and play it
    with open("response.mp3", "wb") as f:
        f.write(audio_data)
    playsound("response.mp3")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
