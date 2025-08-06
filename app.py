# --- IMPORTS ---
import fitz  # PyMuPDF
import numpy as np
import time
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import re
# --- Web server imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
# --- NEW: Import to serve the HTML file ---
from fastapi.responses import FileResponse

# --- 0. OCR CONFIGURATION ---
# On Render, Tesseract will be installed via a build script and be in the system PATH.
# The local Windows path is no longer needed.

# --- 1. API CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # In a production environment, this should be a fatal error.
    raise ValueError("GOOGLE_API_KEY not found. Please set it as an environment variable on Render.")

genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured successfully.")

# --- 2. MODEL INITIALIZATION ---
print("Initializing Gemini models...")
embedding_model_name = 'models/text-embedding-004'
llm_model_name = 'gemini-1.5-flash-latest'
print(f"Using LLM: {llm_model_name}")

embedding_model = genai.GenerativeModel(embedding_model_name)
llm_model = genai.GenerativeModel(llm_model_name)
print("Gemini models initialized.")


# --- 3. FASTAPI APP SETUP ---
app = FastAPI()

# --- Global state (for simplicity) ---
# This simple state is fine for a single-user demo on Render's free tier.
document_state = {
    "text_chunks": None,
    "chunk_embeddings": None,
    "filename": None
}

# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    text: str

# --- 4. CORE LOGIC (Adapted for the web server) ---

def process_and_embed_pdf(pdf_content: bytes, filename: str):
    """Processes the PDF content, chunks, and embeds it."""
    global document_state
    print(f"Processing PDF: {filename}")

    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        full_text = "".join(page.get_text("text") for page in doc)

        if len(full_text.strip()) < 100:
            print("Scanned PDF detected, starting OCR...")
            full_text = ""
            for page_num, page in enumerate(doc):
                print(f"OCR on page {page_num + 1}/{len(doc)}")
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                full_text += pytesseract.image_to_string(img) + "\n"
        doc.close()

        print("Chunking text...")
        raw_text = full_text.replace('-\n', '')
        paragraphs = re.split(r'\n\s*\n', raw_text)
        chunks = [re.sub(r'\s+', ' ', p).strip() for p in paragraphs if len(p.strip()) > 15]

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        print(f"Embedding {len(chunks)} chunks...")
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            response = genai.embed_content(model=embedding_model_name, content=batch)
            all_embeddings.extend(response['embedding'])

        document_state["text_chunks"] = chunks
        document_state["chunk_embeddings"] = np.array(all_embeddings)
        document_state["filename"] = filename
        print("PDF processed successfully.")
        return {"message": f"Successfully processed '{filename}'. Ready to answer questions."}

    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")


# --- 5. API ENDPOINTS ---

# --- NEW: Endpoint to serve the HTML frontend ---
@app.get("/")
async def read_root():
    return FileResponse('index.html')

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload and process a PDF."""
    content = await file.read()
    return process_and_embed_pdf(content, file.filename)

@app.post("/ask/")
async def ask_question(question: Question):
    """Endpoint to ask a question about the processed PDF."""
    if document_state["text_chunks"] is None:
        raise HTTPException(status_code=400, detail="No PDF has been processed yet.")

    try:
        print(f"Received question: {question.text}")
        question_embedding_response = genai.embed_content(model=embedding_model_name, content=question.text)
        question_embedding = np.array(question_embedding_response['embedding']).reshape(1, -1)

        similarities = cosine_similarity(question_embedding, document_state["chunk_embeddings"])[0]
        top_k = 7
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        context = "\n\n---\n\n".join([document_state["text_chunks"][i] for i in top_indices])

        prompt = f"""
        You are a helpful and precise assistant. Your main goal is to answer the user's question accurately based on the provided context.

        **Your Instructions:**
        1.  First, understand the user's question. Are they asking for a specific piece of text (like a "caption" or "title"), or are they asking a general question about the content (like "summarize" or "what is this about")?
        2.  If they ask for a specific piece of text, you MUST provide the full and exact text from the context. Do not summarize it.
        3.  If they ask a general question, you should synthesize the information from the context to provide a comprehensive and helpful answer. You can quote parts of the text if it helps.
        4.  Your knowledge is strictly limited to the context provided. If the answer is not found in the context, you MUST respond with: "I cannot find the answer in this document."
        5.  Do not add any outside information or make up details.

        **CONTEXT FOR THIS TASK:**
        ---
        {context}
        ---

        **User Question:**
        {question.text}

        **Answer:**
        """

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=2048,
            temperature=0.25
        )

        response = llm_model.generate_content(prompt, generation_config=generation_config)

        if response.parts:
            answer = response.text.strip()
        else:
            answer = "The model did not provide a response. This might be due to a safety filter or an issue with the prompt."

        print(f"Generated answer: {answer}")
        return {"answer": answer}

    except Exception as e:
        print(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

# --- 6. RUN THE SERVER ---
if __name__ == "__main__":
    # This allows the script to be run directly for development.
    # For production, you would use a command like: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)