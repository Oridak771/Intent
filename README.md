# Voice-Activated Service Request System

## Overview

The Voice-Activated Service Request System is an innovative application designed to streamline client interactions with organizational services. This application leverages advanced technologies such as Natural Language Processing (NLP), speech recognition (Deepgram), intent classification (fine-tuned DistilRoBERTa), and Named Entity Recognition (Spacy) to efficiently handle voice inputs and convert them into actionable service requests.

## Features

- **Voice Input Capture**: Captures voice input from the user.
- **Speech Recognition**: Transcribes voice input to text using Deepgram's speech recognition technology.
- **Intent Classification**: Classifies the intent of the transcribed text to understand the user's request using a fine-tuned DistilRoBERTa model.
- **Named Entity Recognition**: Extracts necessary entities from the text to complete the service request using Spacy.
- **Service Request Processing**: Processes and fulfills the service request based on the classified intent and extracted entities.

## Architecture

The application's architecture is designed to handle voice inputs efficiently and convert them into actionable service requests. The process involves:

1. Capturing voice input from the user.
2. Transcribing the voice input to text using speech recognition.
3. Classifying the intent of the transcribed text to understand what the user wants to achieve.
4. Extracting necessary entities from the text to complete the service request.
5. Processing and fulfilling the service request based on the classified intent and extracted entities.

## Fine-Tuned Model

The intent classification model used in this application is a fine-tuned version of DistilRoBERTa. The base model was fine-tuned using a custom dataset tailored to specific user intents relevant to the application. You can access the model on [Hugging Face](https://huggingface.co/Oridak771/distilroberta-finetuned).

## Getting Started

To get started with the Voice-Activated Service Request System, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
