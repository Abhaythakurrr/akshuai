import os
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, root_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from better_profanity import profanity
from dotenv import load_dotenv
from prometheus_client import Counter, generate_latest
from starlette.responses import PlainTextResponse
import torch
import numpy as np

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AkshuAI Language Module",
    description="Language module for AkshuAI, providing text generation, embedding, tokenization, and summarization with persona customization and content filtering.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth2 authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Prometheus metrics
requests_counter = Counter("language_requests_total", "Total requests to language module", ["endpoint"])

# Configuration
MODEL_NAME = os.getenv("LANGUAGE_MODEL_NAME", "meta-llama/Llama-3-8B")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 200))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Global model and tokenizer
language_model = None
language_tokenizer = None
text_generator = None

# Pydantic models
class GenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = MAX_TOKENS
    user_id: str = "anonymous"
    session_id: str = "default"
    context: Dict[str, Any] = {}

    @root_validator(pre=True)
    def validate_context(cls, values):
        context = values.get("context", {})
        if "persona" in context and not isinstance(context["persona"], dict):
            raise ValueError("context.persona must be a dictionary")
        return values

class GenerateResponse(BaseModel):
    generated_texts: List[Dict[str, Any]]
    text: str
    finish_reason: str
    token_usage: Dict[str, int] = {}

class EmbedRequest(BaseModel):
    texts: List[str]
    user_id: str = "anonymous"
    session_id: str = "default"

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    token_usage: Dict[str, int] = {}

class TokenizeRequest(BaseModel):
    texts: List[str]
    user_id: str = "anonymous"
    session_id: str = "default"

class TokenizeResponse(BaseModel):
    tokens: List[List[int]]
    token_counts: List[int]

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 100
    user_id: str = "anonymous"
    session_id: str = "default"
    context: Dict[str, Any] = {}

class SummarizeResponse(BaseModel):
    summary: str
    token_usage: Dict[str, int] = {}

def load_model():
    """Load the language model and tokenizer at startup."""
    global language_model, language_tokenizer, text_generator
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        language_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        language_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        text_generator = TextGenerationPipeline(
            model=language_model,
            tokenizer=language_tokenizer,
            device=0 if DEVICE == "cuda" else -1,
            framework="pt",
            max_new_tokens=MAX_TOKENS,
            pad_token_id=language_tokenizer.eos_token_id
        )
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

# Initialize profanity filter
try:
    profanity.load_censor_words()
except Exception as e:
    logger.warning(f"Failed to load profanity filter: {e}")

# Load model at startup
try:
    load_model()
except Exception as e:
    logger.critical(f"Startup failed: {e}")
    raise

def format_persona_instructions(persona: Dict[str, Any]) -> str:
    """Generate prompt instructions based on persona data."""
    if not persona:
        return ""
    
    tone = persona.get("tone", "neutral").lower()
    style = persona.get("style", "general").lower()
    
    instructions = []
    if tone == "formal":
        instructions.append("Respond in a formal and professional tone.")
    elif tone == "casual":
        instructions.append("Use a friendly and casual tone.")
    elif tone == "technical":
        instructions.append("Provide a detailed and technical explanation.")
    
    if style == "concise":
        instructions.append("Keep the response brief and to the point.")
    elif style == "verbose":
        instructions.append("Provide a detailed and comprehensive response.")
    
    return " ".join(instructions) + " " if instructions else ""

def filter_content(text: str) -> tuple[bool, str]:
    """Check text for inappropriate content. Returns (is_clean, message)."""
    # Ensure profanity filter is loaded (can be done once globally)
    try:
        profanity.contains_profanity("test") # Check if loaded, lazy load if needed
    except:
        profanity.load_censor_words()

    if profanity.contains_profanity(text):
        return False, "Text contains inappropriate language."
    sensitive_keywords = ["violence", "hate", "discrimination"]
    for keyword in sensitive_keywords:
        if keyword in text.lower():
            return False, f"Text contains sensitive topic: {keyword}."
    return True, ""

@app.on_event("startup")
async def startup_event():
    logger.info("Language module started.")

@app.get("/")
async def read_root():
    requests_counter.labels(endpoint="/").inc()
    return {"message": "Language module is running"}

@app.get("/health")
async def health_check():
    requests_counter.labels(endpoint="/health").inc()
    if language_model is None or language_tokenizer is None:
        raise HTTPException(status_code=503, detail="Language model is not loaded.")
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    requests_counter.labels(endpoint="/metrics").inc()
    return PlainTextResponse(generate_latest())

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest, token: str = Depends(oauth2_scheme)):
    """Generates text from one or more prompts, applying persona-based tone and content filtering.
    
    Args:
        request: GenerateRequest with prompts, max_tokens, user_id, session_id, and context.
        token: OAuth2 token for authentication.
    
    Returns:
        GenerateResponse with generated texts, finish reason, and token usage.
    
    Raises:
        HTTPException: If model is not loaded (503), input is inappropriate (400), or generation fails (500).
    """
    requests_counter.labels(endpoint="/generate").inc()
    logger.info(f"Received generate request from user {request.user_id}: {len(request.prompts)} prompts")

    if language_model is None or language_tokenizer is None or text_generator is None:
        logger.error("Language model or tokenizer not loaded.")
        raise HTTPException(status_code=503, detail="Language model is not loaded.")

    max_tokens = min(request.max_tokens, MAX_TOKENS)
    persona = request.context.get("persona", {})
    persona_instructions = format_persona_instructions(persona)

    generated_texts = []
    total_input_tokens = 0
    total_output_tokens = 0

    try:
        full_prompts = [f"{persona_instructions}{prompt.replace('<', '<').replace('>', '>')}".strip() for prompt in request.prompts]
        
        # Filter prompts
        for prompt in full_prompts:
            is_clean, message = filter_content(prompt)
            if not is_clean:
                logger.warning(f"Prompt rejected: {message}")
                raise HTTPException(status_code=400, detail=message)

        logger.debug(f"Generating text for {len(full_prompts)} prompts...")
        generation_outputs = text_generator(
            full_prompts,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            pad_token_id=language_tokenizer.eos_token_id,
            batch_size=4
        )

        for i, output in enumerate(generation_outputs):
            generated_text = output["generated_text"].strip()
            
            is_clean, message = filter_content(generated_text)
            if not is_clean:
                logger.warning(f"Generated text rejected: {message}")
                generated_text = "Content filtered due to inappropriate content."
            
            finish_reason = "stop" if len(generated_text) < max_tokens else "length"
            # Re-calculate token usage based on the *generated* text, not including prompt again
            input_tokens_single = len(language_tokenizer.encode(full_prompts[i])) # Tokens for original prompt
            output_tokens_single = len(language_tokenizer.encode(generated_text)) # Tokens for generated text
            
            generated_texts.append({
                "prompt": request.prompts[i], # Store original prompt for reference
                "generated_text": generated_text,
                "finish_reason": finish_reason,
                "token_usage": {
                    "prompt_tokens": input_tokens_single,
                    "completion_tokens": output_tokens_single,
                    "total_tokens": input_tokens_single + output_tokens_single
                }
            })
            
            total_input_tokens += input_tokens_single
            total_output_tokens += output_tokens_single

        text = generated_texts[0]["generated_text"] if generated_texts else ""
        overall_finish_reason = generated_texts[0]["finish_reason"] if generated_texts else "stop"

        response = {
            "generated_texts": generated_texts,
            "text": text,  # For orchestrator compatibility (consider if orchestrator can handle batch?)
            "finish_reason": overall_finish_reason,
            "token_usage": {
                "prompt_tokens": total_input_tokens,
                "completion_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            }
        }

        logger.info(f"Generated {len(generated_texts)} texts. First: {text[:50]}...")
        return response

    except Exception as e:
        logger.error(f"Error during batch text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during text generation: {e}")

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest, token: str = Depends(oauth2_scheme)):
    """Generates embeddings for input texts using the language model.
    
    Args:
        request: EmbedRequest with texts, user_id, and session_id.
        token: OAuth2 token for authentication.
    
    Returns:
        EmbedResponse with embeddings and token usage.
    
    Raises:
        HTTPException: If model is not loaded (503) or embedding fails (500).WindowState.Normal
    """
    requests_counter.labels(endpoint="/embed").inc()
    logger.info(f"Received embed request from user {request.user_id}: {len(request.texts)} texts")

    # For embeddings, using a dedicated Sentence Transformer model is generally better
    # However, for demonstration, we will use the loaded language model's embeddings

    if language_model is None or language_tokenizer is None:
        logger.error("Language model or tokenizer not loaded.")
        raise HTTPException(status_code=503, detail="Language model is not loaded.")

    try:
        # Filter input texts
        for text in request.texts:
            is_clean, message = filter_content(text)
            if not is_clean:
                logger.warning(f"Text rejected: {message}")
                raise HTTPException(status_code=400, detail=message)

        # Tokenize and generate embeddings
        inputs = language_tokenizer(
            request.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            # Get the last hidden states from the model
            outputs = language_model(**inputs, output_hidden_states=True)
            # Use the last hidden state, mean-pooled over non-padding tokens
            # Need to handle attention mask for proper pooling
            last_hidden_states = outputs.hidden_states[-1]
            # Create a mask for non-padding tokens
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
            # Apply mask and sum, then divide by the number of non-padding tokens
            sum_embeddings = torch.sum(last_hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9) # Avoid division by zero
            embeddings = (sum_embeddings / sum_mask).cpu().numpy().tolist()

        token_usage = {
            "prompt_tokens": sum(len(language_tokenizer.encode(text)) for text in request.texts), # Count tokens in original texts
            "completion_tokens": 0, # Embeddings don't generate new tokens in this sense
            "total_tokens": sum(len(language_tokenizer.encode(text)) for text in request.texts)
        }

        response = {
            "embeddings": embeddings,
            "token_usage": token_usage
        }

        logger.info(f"Generated {len(embeddings)} embeddings.")
        return response

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during embedding generation: {e}")

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest, token: str = Depends(oauth2_scheme)):
    """Tokenizes input texts using the model's tokenizer.
    
    Args:
        request: TokenizeRequest with texts, user_id, and session_id.
        token: OAuth2 token for authentication.
    
    Returns:
        TokenizeResponse with token IDs and counts.
    
    Raises:
        HTTPException: If tokenizer is not loaded (503) or tokenization fails (500).
    """
    requests_counter.labels(endpoint="/tokenize").inc()
    logger.info(f"Received tokenize request from user {request.user_id}: {len(request.texts)} texts")

    if language_tokenizer is None:
        logger.error("Tokenizer not loaded.")
        raise HTTPException(status_code=503, detail="Tokenizer is not loaded.")

    try:
        tokens = [language_tokenizer.encode(text, add_special_tokens=True) for text in request.texts]
        token_counts = [len(t) for t in tokens]

        response = {
            "tokens": tokens,
            "token_counts": token_counts
        }

        logger.info(f"Tokenized {len(tokens)} texts.")
        return response

    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        raise HTTPException(status_code=500, detail=f"Error during tokenization: {e}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest, token: str = Depends(oauth2_scheme)):
    """Summarizes input text using the language model.
    
    Args:
        request: SummarizeRequest with text, max_length, user_id, session_id, and context.
        token: OAuth2 token for authentication.
    
    Returns:
        SummarizeResponse with summary and token usage.
    
    Raises:
        HTTPException: If model is not loaded (503), input is inappropriate (400), or summarization fails (500).
    """
    requests_counter.labels(endpoint="/summarize").inc()
    logger.info(f"Received summarize request from user {request.user_id}: {request.text[:50]}...")

    if language_model is None or language_tokenizer is None or text_generator is None:
        logger.error("Language model or tokenizer not loaded.")
        raise HTTPException(status_code=503, detail="Language model is not loaded.")

    # Filter input text
    is_clean, message = filter_content(request.text)
    if not is_clean:
        logger.warning(f"Text rejected: {message}")
        raise HTTPException(status_code=400, detail=message)

    max_length = min(request.max_length, MAX_TOKENS) # Use MAX_TOKENS as an upper bound for summary length
    persona = request.context.get("persona", {})
    persona_instructions = format_persona_instructions(persona)

    try:
        # Prepare summarization prompt
        # Adding persona instructions to the summarization prompt
        prompt = f"{persona_instructions}Summarize the following text in no more than {max_length} tokens: {request.text}"
        
        logger.debug(f"Summarization prompt: {prompt[:100]}...")

        # Use the text_generator pipeline for summarization as well
        # Using non-deterministic sampling might give more natural summaries, but deterministic is often preferred.
        # Set temperature to a low value for more focused generation.
        generation_output = text_generator(
            prompt,
            max_new_tokens=max_length,
            do_sample=False,  # Deterministic for summarization
            temperature=0.1, # Lower temperature for more focused summary
            top_p=1.0,
            return_full_text=False,
            pad_token_id=language_tokenizer.eos_token_id
            # TODO: Add stop sequence handling during generation if needed for summarization
        )

        summary = generation_output[0]["generated_text"].strip()
        
        is_clean, message = filter_content(summary)
        if not is_clean:
            logger.warning(f"Summary rejected: {message}")
            summary = "Summary filtered due to inappropriate content."

        # Calculate token usage for the summarization task
        input_tokens = len(language_tokenizer.encode(prompt)) # Tokens for the full prompt (instruction + text)
        output_tokens = len(language_tokenizer.encode(summary))
        token_usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

        response = {
            "summary": summary,
            "token_usage": token_usage
        }

        logger.info(f"Generated summary: {summary[:50]}...")
        return response

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Error during summarization: {e}")

# TODO: Add other endpoints as defined by the architecture (e.g., translation, sentiment analysis).
# Ensure these endpoints are also updated to handle context, persona, and content filtering if relevant.
