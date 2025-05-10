# akshu-ai/services/persona/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI()

# Placeholder for persona storage (in-memory dictionary - NOT for production)
personas_db: Dict[str, Dict[str, Any]] = {}

class PersonaCreate(BaseModel):
    user_id: str
    name: str
    preferences: Dict[str, Any] = {}
    tone: str = "neutral"
    privacy_settings: Dict[str, Any] = {}

class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    tone: Optional[str] = None
    privacy_settings: Optional[Dict[str, Any]] = None

class PersonaResponse(BaseModel):
    user_id: str
    name: str
    preferences: Dict[str, Any]
    tone: str
    privacy_settings: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Persona service is running"}

@app.post("/create", response_model=PersonaResponse)
async def create_persona(persona: PersonaCreate):
    print(f"Received request to create persona for user: {persona.user_id}")
    # TODO: Implement actual storage in a database (e.g., PostgreSQL)
    # - Check if user_id already exists
    # - Store the new persona data

    if persona.user_id in personas_db:
        raise HTTPException(status_code=400, detail="Persona for this user_id already exists")

    persona_data = persona.dict()
    personas_db[persona.user_id] = persona_data
    print(f"Created persona for user: {persona.user_id}")
    return PersonaResponse(**persona_data)

@app.get("/get/{user_id}", response_model=PersonaResponse)
async def get_persona(user_id: str):
    print(f"Received request to get persona for user: {user_id}")
    # TODO: Implement actual retrieval from a database

    if user_id not in personas_db:
        raise HTTPException(status_code=404, detail="Persona not found")

    persona_data = personas_db[user_id]
    print(f"Retrieved persona for user: {user_id}")
    return PersonaResponse(**persona_data)

@app.put("/update/{user_id}", response_model=PersonaResponse)
async def update_persona(user_id: str, persona_update: PersonaUpdate):
    print(f"Received request to update persona for user: {user_id}")
    # TODO: Implement actual update in a database
    # - Check if user_id exists
    # - Update the persona data with non-None fields from persona_update

    if user_id not in personas_db:
        raise HTTPException(status_code=404, detail="Persona not found")

    current_persona_data = personas_db[user_id]
    update_data = persona_update.dict(exclude_unset=True)

    for key, value in update_data.items():
        current_persona_data[key] = value

    personas_db[user_id] = current_persona_data
    print(f"Updated persona for user: {user_id}")
    return PersonaResponse(**current_persona_data)

# TODO: Add endpoint for deleting a persona
