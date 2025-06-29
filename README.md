# AI Policy Assistant Chatbot with RAG, Speech-to-Text (STT) & Text-to-Speech (TTS)

This project implements an intelligent chatbot designed to answer user questions based on specific company policy documents. It leverages Azure AI Services for powerful natural language processing and speech capabilities, combined with a Retrieval-Augmented Generation (RAG) architecture to provide accurate and contextual responses.

The chatbot features a user-friendly web interface with both text and voice interaction, supporting multiple languages.

## Features

* **Retrieval-Augmented Generation (RAG):** Utilizes FAISS (Facebook AI Similarity Search) and HuggingFace embeddings to retrieve relevant information from pre-indexed company policy documents, ensuring answers are grounded in provided context.
* **Azure OpenAI Integration:** Processes user queries and generates comprehensive responses using Azure OpenAI's language models.
* **Multilingual Support:** Communicate with the chatbot in English (`en-US`), Bahasa Indonesia (`id-ID`), and Chinese (`zh-CN`).
* **Real-time Speech-to-Text (STT):** Transcribes user's spoken input into text *as they speak* using the browser's native Web Speech API, providing an immediate and interactive experience.
* **Azure Text-to-Speech (TTS):** Synthesizes the chatbot's text responses into natural-sounding speech using high-quality Azure AI Speech neural voices.
* **Intuitive Web Interface:** A clean and responsive web interface built with HTML, JavaScript, and Tailwind CSS.
* **Scalable Backend:** Flask-based backend capable of handling API requests for chat, STT (if needed for batch processing), and TTS.

## Technologies Used

### Backend (Python/Flask)

* **Python 3.x**
* **Flask:** Web framework for the API endpoints.
* **`openai`:** Python client for Azure OpenAI Service.
* **`azure-identity`:** For Azure Active Directory (Entra ID) authentication with Azure services.
* **`azure-cognitiveservices-speech`:** Azure AI Speech SDK for STT and TTS.
* **`langchain-community`:** For FAISS vector store integration.
* **`langchain-huggingface`:** For `HuggingFaceEmbeddings` (specifically "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").


### Frontend (Web)

* **HTML5, CSS (Tailwind CSS):** Structure and styling.
* **JavaScript:** For dynamic interaction, API calls, and integration with browser's Web Speech API.
* **Web Speech API:** Browser-native API for real-time Speech-to-Text input.
* **Font Awesome:** For icons (microphone, play/pause, etc.).

### Azure AI Services

* **Azure OpenAI Service:** With a deployed model (e.g., `gpt-35-turbo`, `gpt-4`) configured for chat completions.
* **Azure AI Speech Service:** For Speech-to-Text and Text-to-Speech capabilities.

### Vector Store

* **FAISS:** Used to store and retrieve embeddings of the company policy documents.

## Project Structure

├── app.py                  # Flask backend application
├── index.html              # Frontend web interface
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables file
├── vectorstore/            # Directory for FAISS index (e.g., rekso_2023_2025_v2)
│   └── (FAISS index files)


## Setup and Installation

### Prerequisites

1.  **Python 3.x:** Installed on your system.
2.  **Azure Subscription:**
    * **Azure OpenAI Service:** Create a resource and deploy a chat completion model (e.g., `gpt-35-turbo`, `gpt-4`). Note its `Endpoint` and `Deployment name`.
    * **Azure AI Speech Service:** Create a resource. Note its `Key` and `Region`.
3.  **FFmpeg:**
    * `pydub` (used in backend for robust STT audio handling) requires `ffmpeg` to be installed and available in your system's PATH.
    * **Windows:** Download binaries from [gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your system's PATH.
    * **macOS:** `brew install ffmpeg`
    * **Linux (Ubuntu/Debian):** `sudo apt-get update && sudo apt-get install ffmpeg`
4.  **Company Policy Data:** Ensure you have processed your company policy documents and created the FAISS vector store located at `vectorstore/rekso_2023_2025_v2`. This project assumes this vector store is already available.

### Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-project-directory>
    ```

2.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Python dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    Flask
    Flask-Cors
    openai
    azure-identity
    langchain-community
    langchain-huggingface
    pydub
    sentence-transformers
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    Create a file named `.env` in the root of your project directory, based on `.env.example`, and fill in your Azure credentials:

    ```ini
    AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
    AZURE_OPENAI_COMPLETION_DEPLOYMENT="YOUR_OPENAI_DEPLOYMENT_NAME"
    API_VERSION="2024-02-15-preview" # Or latest stable API version you are using
    SPEECH_KEY="YOUR_AZURE_SPEECH_KEY"
    SPEECH_REGION="YOUR_AZURE_SPEECH_REGION"
    ```
    (Note: `API_VERSION` might need adjustment based on your Azure OpenAI deployment).

6.  **Prepare Directories:**
    Ensure the `vectorstore/rekso_2023_2025_v2` directory exists and contains your FAISS index files.
    Create an `uploads` directory at the root of your project:
    ```bash
    mkdir uploads
    ```

7.  **Run the Flask Backend:**
    ```bash
    python app.py
    ```
    The backend server will typically run on `http://127.0.0.1:5000`.

### Frontend Setup

The frontend is a simple HTML, CSS, and JavaScript file. No complex build steps are required.

1.  **Serve the `index.html` file:**
    To enable microphone access and avoid potential CORS issues when opening `index.html` directly, it's best to serve it via a simple local web server.
    Open a *new* terminal or command prompt (separate from where your Flask app is running) in your project's root directory:
    ```bash
    python -m http.server
    ```
    This will serve the files from the current directory, usually on `http://localhost:8000`.

## Usage

1.  **Ensure both the Flask backend and the local frontend server are running.**
2.  **Open your web browser** and navigate to `http://localhost:8000/index.html` (or the address provided by `python -m http.server`).
3.  **Select Your Preferred Language:** An overlay will appear prompting you to choose between English, Bahasa Indonesia, or Chinese. Select one to proceed.
4.  **Start Chatting:**
    * **Text Input:** Type your question into the input field and click the "Send" button.
    * **Voice Input (STT):** Click the microphone icon. As you speak, your words will appear in the input field in real-time. Click the microphone again (or stop speaking for a few seconds) to stop recording, and your transcribed message will be sent automatically.
5.  **Listen to Responses (TTS):** When the chatbot responds, a "play" icon (volume-up) will appear next to its message. Click this icon to listen to the chatbot's response in the selected language.