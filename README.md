📚 Chatbot Agent - Company Policy
This Flask-based chatbot is designed to provide quick and accurate answers to questions regarding PT Rekso Nasional Food's company policies for the years 2023–2025. It achieves this by leveraging Azure OpenAI for conversational AI and Retrieval-Augmented Generation (RAG) to fetch relevant information directly from official PDF policy documents. The documents are processed into a FAISS vector database using HuggingFace multilingual embeddings for efficient retrieval.

✨ Features
Azure OpenAI Chat API (GPT-4o): Utilizes the advanced GPT-4o model for highly accurate and contextual natural language understanding.

Retrieval-Augmented Generation (RAG) with FAISS: Enhances chatbot responses by retrieving relevant policy clauses from the stored vector database, ensuring answers are grounded in official documents.

Multilingual Embedding Model: Integrates HuggingFace's paraphrase-multilingual-MiniLM-L12-v2 to support policy inquiries in English, Bahasa Indonesia, and Chinese.

Local FAISS Vectorstore: Policy document embeddings are stored locally, enabling fast and efficient information retrieval without external database dependencies.

Flask API Endpoint /chat with CORS: Provides a flexible and accessible RESTful API for integrating the chatbot into various frontend applications, with CORS enabled for broad compatibility.

Multi-language Support: Designed to understand and respond to queries in English, Bahasa Indonesia, and Chinese, catering to a diverse workforce.

📂 Project Structure
project-root/
├── data/
│   └── peraturan-rekso.pdf    # Source PDF document containing company policies
├── vectorstore/
│   └── rekso_2023_2025/       # Directory to store the FAISS vector database
├── app.py                     # Flask application backend for the chatbot API
├── rag_builder.py             # Script to process PDFs and build the FAISS vectorstore
├── templates/
│   └── index.html             # Basic HTML frontend web client for interaction
├── requirements.txt           # Python dependencies for the project
└── README.md                  # This README file

🚀 Installation & Setup
Follow these steps to get the chatbot up and running on your local machine.

1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

(Replace your-username/your-repo-name.git with your actual repository URL)

2. Create and Activate a Virtual Environment (Recommended)
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Python Dependencies
With your virtual environment activated, install all required Python libraries:

pip install -r requirements.txt

requirements.txt content:
flask
flask-cors
openai
azure-identity
langchain
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
python-dotenv

4. Prepare Policy Document
Place your company policy PDF document (e.g., peraturan-rekso.pdf) into the data/ directory.

5. Set Up Azure OpenAI Credentials
Create a .env file in the project-root/ directory and add your Azure OpenAI service details.

AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
AZURE_OPENAI_DEPLOYMENT_NAME="your_gpt4o_deployment_name" # e.g., gpt-4o-deployment

(Ensure your_gpt4o_deployment_name matches the deployment name of your GPT-4o model on Azure OpenAI Studio.)

6. Build the FAISS Vectorstore
Run the rag_builder.py script to process the PDF document and create the FAISS vector database. This may take some time depending on the size of your PDF.

python rag_builder.py

This will create the rekso_2023_2025 directory inside vectorstore/.

▶️ Running the Chatbot
1. Start the Flask Backend
From the project-root/ directory, run the Flask application:

python app.py

The Flask backend will typically run on http://127.0.0.1:5000.

2. Access the Frontend
Open your web browser and navigate to http://127.0.0.1:5000/ (or the address where your Flask app is hosted). This will load the index.html frontend client, allowing you to interact with the chatbot.

⚙️ Usage
Once the frontend is loaded, you can type your questions related to PT Rekso Nasional Food's 2023–2025 company policies into the chat interface. The chatbot will retrieve relevant information from the PDF and generate an answer using Azure OpenAI.

Example Questions:
"What is the company's policy on remote work?"

"Cuti tahunan berapa hari?" (How many days is annual leave?)

"公司关于出差报销的规定是什么？" (What are the company's regulations on travel expense reimbursement?)

🤝 Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.