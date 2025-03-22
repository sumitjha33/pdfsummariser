from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from pypdf import PdfReader
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
from googletrans import Translator
import requests
import time
import io
import pyttsx3
from gtts import gTTS
import tempfile
import hashlib
from urllib.parse import quote
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

app = FastAPI(title="PDF Summarizer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

LANGUAGES = {
    "english": {
        "prompt": "Respond in English language only.",
        "code": "en"
    },
    "hindi": {
        "prompt": "केवल हिंदी भाषा में उत्तर दें।",
        "code": "hi"
    },
    "kannada": {
        "prompt": "ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಮಾತ್ರ ಪ್ರತಿಕ್ರಿಯಿಸಿ.",
        "code": "kn"
    }
}

DEPTH_LEVELS = {
    "short": {
        "instruction": """
        Provide a concise summary in 2-3 short paragraphs.
        - Keep sentences brief (15-20 words max)
        - Total length should be around 150 words
        - Break into 2-3 clear paragraphs
        - Use simple language and short sentences
        """,
        "percentage": 20
    },
    "detailed": {
        "instruction": """
        Provide a balanced explanation in 4-5 medium paragraphs.
        - Keep sentences moderate (20-25 words max)
        - Total length should be around 300 words
        - Break into 4-5 well-structured paragraphs
        - Use clear examples and explanations
        """,
        "percentage": 40
    },
    "deep_dive": {
        "instruction": """
        Provide a comprehensive explanation in 6-7 detailed paragraphs.
        - Keep sentences informative but clear (25-30 words max)
        - Total length should be around 500 words
        - Break into 6-7 detailed paragraphs
        - Include thorough examples and explanations
        - Add relevant context and connections
        """,
        "percentage": 80
    }
}

AGE_GROUPS = {
    "kids": {
        "range": "0-9",
        "instruction": "Explain in very simple terms, use fun examples, and child-friendly language. Avoid complex words."
    },
    "teens": {
        "range": "9-18",
        "instruction": "Use engaging examples relevant to teenagers, explain concepts clearly, and maintain an interesting tone."
    },
    "adults": {
        "range": "18+",
        "instruction": "Provide detailed explanations with mature examples and professional terminology where appropriate."
    }
}

PROMPT_TEMPLATES = {
    "podcast": """
    Present this content like a {age_group} podcast:
    {depth_instruction}
    {language_instruction}
    {age_instruction}
    
    Important Instructions:
    - Focus on key points and main ideas
    - Adapt language and examples for {age_range} age group
    - Keep it engaging and appropriate for the age level
    
    Content to analyze: {{text}}
    """,
    
    "entertainment": """
    Transform this content into an entertaining and engaging format:
    {depth_instruction}
    {language_instruction}
    {age_instruction}
    - Focus on key points and main ideas
    - Adapt language and examples for {age_range} age group
    - Use a light, engaging tone
    - Include interesting examples and analogies
    - Highlight fascinating aspects
    - Keep it fun and accessible
    - Add elements of surprise or interest
    - Aim for 4-5 minute read time

    Content to analyze: {{text}}
    """,
    
    "educational": """
    Present this content in a comprehensive educational format:
    {depth_instruction}
    {language_instruction}
    {age_instruction}
    - Focus on key points and main ideas
    - Adapt language and examples for {age_range} age group
    - Start with clear learning objectives
    - Explain concepts thoroughly with examples
    - Include definitions of technical terms
    - Provide step-by-step explanations
    - Add key points for review
    - Connect ideas to practical applications
    - Use academic but accessible language

    Content to analyze: {{text}}
    """,
    
    "storytelling": """
    Transform this content into a narrative story format:
    {depth_instruction}
    {language_instruction}
    {age_instruction}
    - Focus on key points and main ideas
    - Adapt language and examples for {age_range} age group
    - Create a flowing narrative arc
    - Use descriptive language
    - Add character elements where appropriate
    - Build connections between ideas
    - Include a clear beginning, middle, and end
    - Make it engaging and memorable
    - Keep it concise but impactful

    Content to analyze: {{text}}
    """
}

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file as a single document."""
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n\n"
    return full_text.strip()

def split_into_sentences(text):
    """Split text into sentences using regex."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def chunk_text(text, max_length=1000):
    """Split text into chunks of max_length while preserving sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_speech(text, use_offline=True):
    """Generate speech using pyttsx3 (offline) or gTTS (online)"""
    try:
        if use_offline:
            # Try offline TTS first
            engine = pyttsx3.init()
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            engine.save_to_file(text, temp_file.name)
            engine.runAndWait()
            return temp_file.name
        else:
            # Fallback to gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tts.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise

def format_response(text):
    """Format the response text with proper paragraph breaks and sentence lengths"""
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Format each paragraph
    formatted_paragraphs = []
    for paragraph in paragraphs:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        # Add line break after each sentence
        formatted_sentences = '\n'.join(sentences)
        formatted_paragraphs.append(formatted_sentences)
    
    # Join paragraphs with double line breaks
    return '\n\n'.join(formatted_paragraphs)

class TextRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str
    sentences: List[str]
    language: str

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_pdf(
    file: UploadFile = File(...),
    mode: str = Form("educational"),
    depth: str = Form("detailed"),
    language: str = Form("english"),
    age_group: str = Form("teens")
):
    """API endpoint to summarize PDF with options"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        text = extract_text_from_pdf(file.file)
        depth_config = DEPTH_LEVELS.get(depth, DEPTH_LEVELS["detailed"])
        age_config = AGE_GROUPS.get(age_group)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=2048,
        )
        
        base_template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["educational"])
        lang_config = LANGUAGES.get(language, LANGUAGES["english"])
        
        prompt_text = f"{lang_config['prompt']}\n\n" + base_template.format(
            depth_instruction=depth_config["instruction"],
            language_instruction="",
            percentage=depth_config["percentage"],
            age_instruction=age_config["instruction"],
            age_group=age_group,
            age_range=age_config["range"]
        )
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template=prompt_text
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text)
        
        response = format_response(response)
        
        if language != "english":
            translator = Translator()
            response = translator.translate(response, dest=lang_config["code"]).text
        
        response = response.replace('*', '').replace('#', '')
        sentences = split_into_sentences(response)
        
        return SummaryResponse(
            summary=response,
            sentences=sentences,
            language=language
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech")
async def generate_speech(request: TextRequest):
    """Generate speech from text"""
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        tts = gTTS(text=request.text, lang='en', tld='ca', lang_check=False, slow=False)
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tts.save(temp_file.name)
        
        return StreamingResponse(
            io.BytesIO(temp_file.read()),
            media_type="audio/mp3",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image")
async def generate_image(request: TextRequest):
    """Get relevant image for text"""
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
        
    try:
        search_query = quote(request.text[:50])
        url = f"https://api.unsplash.com/photos/random?query={search_query}&orientation=landscape"
        
        headers = {
            "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
        }
        
        response = requests.get(url, headers=headers)
        if response.ok:
            data = response.json()
            return {
                "image_url": data['urls']['regular'],
                "alt": data.get('alt_description', 'Generated image'),
                "credit": data['user']['name']
            }
            
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
