import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

#  Model selection
HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

def load_llm_chat():
    print(f" Step 1: Connecting to Chat Interface ({HUGGINGFACE_REPO_ID})...")
    
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational", 
        temperature=0.5,
    )
    # Wrapping it for Chat compatibility
    return ChatHuggingFace(llm=llm)

# Step 2: Database Loading
print(" Step 2: Loading FAISS Database...")
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Prompt & Chain

CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the user's question.
If the answer is not in the context, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Answer:
(Provide the answer here)

 DISCLAIMER: This information is provided for educational purposes only. Please consult a certified doctor or medical professional before starting any treatment or medication.
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE, 
    input_variables=["context", "question"]
)



qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm_chat(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

# Step 4: Run
user_query = input("\n💬 Sawal likhein: ")
print("🔍 Searching context and generating answer...")

try:
    response = qa_chain.invoke({'query': user_query})
    print("\n RESULT: ", response["result"])
    print("\n SOURCE DOCUMENTS: ", [doc.metadata for doc in response["source_documents"]])
except Exception as e:
    print(f" Error logic update needed: {e}")