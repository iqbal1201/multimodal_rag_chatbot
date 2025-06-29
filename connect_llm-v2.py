import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# --------- ENV CONFIGURATION ---------
ENDPOINT = os.getenv("ENDPOINT_URL", "https://chatbot-ai-new.openai.azure.com/")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
API_VERSION = os.getenv("API_VERSION", "2025-01-01-preview")


# --------- AZURE OPENAI CLIENT WITH ENTRA ID ---------
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION,
)

# --------- CUSTOM PROMPT TEMPLATE ---------
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

prompt_template = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

# --------- VECTORSTORE + EMBEDDINGS (FAISS) ---------
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Function to query Azure OpenAI with context
def query_azure_openai(context_docs, question):
    context_text = "\n".join([doc.page_content for doc in context_docs])
    filled_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=question)

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": filled_prompt}],
        temperature=0.5,
        max_tokens=800
    )

    return response.choices[0].message.content, context_docs

# Run QA chain manually
user_query = input("Write Query Here: ")
retrieved_docs = retriever.invoke(user_query)

# Query LLM with context
answer, source_documents = query_azure_openai(retrieved_docs, user_query)

print("\nRESULT:\n", answer)
print("\nSOURCE DOCUMENTS:\n")
for doc in source_documents:
    print(f"- {doc.metadata.get('source')}: {doc.page_content[:200]}...\n")
