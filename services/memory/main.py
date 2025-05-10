import os
import json
import loggingrom typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Dependsrom fastapi.security import OAuth2PasswordBearerrom pydantic import BaseModel
import redis
from dotenv import load_dotenvrom prometheus_client import Counter, generate_latestrom starlette.responses import PlainTextResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AkshuAI Memory Module",
    description="Memory module for AkshuAI, providing persistent storage and retrieval of user data, embeddings, and context.",
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
requests_counter = Counter("memory_requests_total", "Total requests to memory module", ["endpoint"])

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True
)

# Configuration
MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", 604800))  # 7 days default

# Pydantic models
class StoreRequest(BaseModel):
    user_id: str
    session_id: str
    key: str
    value: Any
    ttl: int = MEMORY_TTL_SECONDS

class RetrieveRequest(BaseModel):
    user_id: str
    session_id: str
    key: str

class UpdateRequest(BaseModel):
    user_id: str
    session_id: str
    key: str
    value: Any
    ttl: int = MEMORY_TTL_SECONDS

class DeleteRequest(BaseModel):
    user_id: str
    session_id: str
    key: str

class ListKeysRequest(BaseModel):
    user_id: str
    session_id: str

class StoreResponse(BaseModel):
    status: str
    key: str

class RetrieveResponse(BaseModel):
    key: str
    value: Any

class UpdateResponse(BaseModel):
    status: str
    key: str

class DeleteResponse(BaseModel):
    status: str
    key: str

class ListKeysResponse(BaseModel):
    keys: List[str]

def get_redis_key(user_id: str, session_id: str, key: str) -> str:
    """Generate a unique Redis key for the user and session."""
    return f"memory:{user_id}:{session_id}:{key}"

@app.on_event("startup")
async def startup_event():
    """Verify Redis connection and log startup."""
    try:
        redis_client.ping()
        logger.info("Memory module started. Connected to Redis.")
    except redis.ConnectionError as e:
        logger.critical(f"Failed to connect to Redis: {e}")
        raise RuntimeError(f"Failed to connect to Redis: {e}")

@app.get("/")
async def read_root():
    requests_counter.labels(endpoint="/").inc()
    return {"message": "Memory module is running"}

@app.get("/health")
async def health_check():
    requests_counter.labels(endpoint="/health").inc()
    try:
        redis_client.ping()
        return {"status": "healthy"}
    except redis.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis connection failed")

@app.get("/metrics")
async def metrics():
    requests_counter.labels(endpoint="/metrics").inc()
    return PlainTextResponse(generate_latest())

@app.post("/store", response_model=StoreResponse)
async def store(request: StoreRequest, token: str = Depends(oauth2_scheme)):
    """Store a key-value pair in memory for a user and session.
    
    Args:
        request: StoreRequest with user_id, session_id, key, value, and ttl.
        token: OAuth2 token for authentication.
    
    Returns:
        StoreResponse with status and key.
    
    Raises:
        HTTPException: If storage fails (500).
    """
    requests_counter.labels(endpoint="/store").inc()
    logger.info(f"Store request from user {request.user_id}: key={request.key}")

    try:
        redis_key = get_redis_key(request.user_id, request.session_id, request.key)
        redis_client.setex(redis_key, request.ttl, json.dumps(request.value))
        logger.info(f"Stored key {request.key} for user {request.user_id}")
        return {"status": "stored", "key": request.key}
    except Exception as e:
        logger.error(f"Error storing key {request.key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing data: {e}")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest, token: str = Depends(oauth2_scheme)):
    """Retrieve a value from memory by key for a user and session.
    
    Args:
        request: RetrieveRequest with user_id, session_id, and key.
        token: OAuth2 token for authentication.
    
    Returns:
        RetrieveResponse with key and value.
    
    Raises:
        HTTPException: If key is not found (404) or retrieval fails (500).
    """
    requests_counter.labels(endpoint="/retrieve").inc()
    logger.info(f"Retrieve request from user {request.user_id}: key={request.key}")

    try:
        redis_key = get_redis_key(request.user_id, request.session_id, request.key)
        value = redis_client.get(redis_key)
        if value is None:
            logger.warning(f"Key {request.key} not found for user {request.user_id}")
            raise HTTPException(status_code=404, detail="Key not found")
        logger.info(f"Retrieved key {request.key} for user {request.user_id}")
        return {"key": request.key, "value": json.loads(value)}
    except Exception as e:
        logger.error(f"Error retrieving key {request.key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {e}")

@app.post("/update", response_model=UpdateResponse)
async def update(request: UpdateRequest, token: str = Depends(oauth2_scheme)):
    """Update a key-value pair in memory for a user and session.
    
    Args:
        request: UpdateRequest with user_id, session_id, key, value, and ttl.
        token: OAuth2 token for authentication.
    
    Returns:
        UpdateResponse with status and key.
    
    Raises:
        HTTPException: If key is not found (404) or update fails (500).
    """
    requests_counter.labels(endpoint="/update").inc()
    logger.info(f"Update request from user {request.user_id}: key={request.key}")

    try:
        redis_key = get_redis_key(request.user_id, request.session_id, request.key)
        if not redis_client.exists(redis_key):
            logger.warning(f"Key {request.key} not found for user {request.user_id}")
            raise HTTPException(status_code=404, detail="Key not found")
        redis_client.setex(redis_key, request.ttl, json.dumps(request.value))
        logger.info(f"Updated key {request.key} for user {request.user_id}")
        return {"status": "updated", "key": request.key}
    except Exception as e:
        logger.error(f"Error updating key {request.key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating data: {e}")

@app.post("/delete", response_model=DeleteResponse)
async def delete(request: DeleteRequest, token: str = Depends(oauth2_scheme)):
    """Delete a key from memory for a user and session.
    
    Args:
        request: DeleteRequest with user_id, session_id, and key.
        token: OAuth2 token for authentication.
    
    Returns:
        DeleteResponse with status and key.
    
    Raises:
        HTTPException: If key is not found (404) or deletion fails (500).
    """
    requests_counter.labels(endpoint="/delete").inc()
    logger.info(f"Delete request from user {request.user_id}: key={request.key}")

    try:
        redis_key = get_redis_key(request.user_id, request.session_id, request.key)
        if not redis_client.exists(redis_key):
            logger.warning(f"Key {request.key} not found for user {request.user_id}")
            raise HTTPException(status_code=404, detail="Key not found")
        redis_client.delete(redis_key)
        logger.info(f"Deleted key {request.key} for user {request.user_id}")
        return {"status": "deleted", "key": request.key}
    except Exception as e:
        logger.error(f"Error deleting key {request.key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting data: {e}")

@app.post("/list_keys", response_model=ListKeysResponse)
async def list_keys(request: ListKeysRequest, token: str = Depends(oauth2_scheme)):
    """List all keys stored for a user and session.
    
    Args:
        request: ListKeysRequest with user_id and session_id.
        token: OAuth2 token for authentication.
    
    Returns:
        ListKeysResponse with list of keys.
    
    Raises:
        HTTPException: If listing fails (500).
    """
    requests_counter.labels(endpoint="/list_keys").inc()
    logger.info(f"List keys request from user {request.user_id}")

    try:
        pattern = get_redis_key(request.user_id, request.session_id, "*")
        keys = redis_client.keys(pattern)
        # Extract key names from full Redis keys
        key_names = [key.split(":", 3)[-1] for key in keys]
        logger.info(f"Listed {len(key_names)} keys for user {request.user_id}")
        return {"keys": key_names}
    except Exception as e:
        logger.error(f"Error listing keys: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing keys: {e}")