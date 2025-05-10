# akshu-ai/services/reasoning/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import spacy
import logging

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model (fallback to download if missing)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading en_core_web_sm model for spaCy...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# === Models ===

class TaskInput(BaseModel):
    task_description: str
    context: Dict[str, Any] = {}

class PlanStep(BaseModel):
    step_description: str
    module: str
    action: str
    parameters: Dict[str, Any] = {}

class PlanResponse(BaseModel):
    plan: List[PlanStep]
    status: str = "success"
    message: str = "Plan generated successfully"

# === Helper Logic ===

def classify_task(task_description: str) -> str:
    """Classifies the task into an intent category."""
    doc = nlp(task_description.lower())

    if "capital" in task_description and "population" in task_description:
        return 'qa_population_and_capital'
    elif "recipe" in task_description:
        return 'recipe_search'
    elif "summarize" in task_description:
        return 'summarization'
    elif "what is" in task_description:
        return 'qa_fact_lookup'
    elif "remember" in task_description:
        return 'memory_store'
    elif "open" in task_description or "execute" in task_description:
        return 'execution_task'
    return 'unknown'

def generate_plan(task_description: str) -> List[PlanStep]:
    """Generates a plan based on the classified intent."""
    intent = classify_task(task_description)
    logger.info(f"Classified intent: {intent}")
    plan_steps: List[PlanStep] = []

    if intent == 'qa_population_and_capital':
        plan_steps = [
            PlanStep(
                step_description="Find the capital of France",
                module="language",
                action="/question_answering",
                parameters={"question": "What is the capital of France?"}
            ),
            PlanStep(
                step_description="Find the population of France",
                module="language",
                action="/question_answering",
                parameters={"question": "What is the population of France?"}
            )
        ]

    elif intent == 'recipe_search':
        plan_steps = [
            PlanStep(
                step_description="Search for a chicken soup recipe",
                module="language",
                action="/web_search",
                parameters={"query": "chicken soup recipe"}
            ),
            PlanStep(
                step_description="Summarize the recipe",
                module="language",
                action="/summarize",
                parameters={"text": "<PLACEHOLDER_RECIPE_TEXT>"}
            )
        ]

    elif intent == 'summarization':
        plan_steps = [
            PlanStep(
                step_description="Summarize the input text",
                module="language",
                action="/summarize",
                parameters={"text": task_description}
            )
        ]

    elif intent == 'qa_fact_lookup':
        plan_steps = [
            PlanStep(
                step_description="Answer the user's fact-based question",
                module="language",
                action="/question_answering",
                parameters={"question": task_description}
            )
        ]

    elif intent == 'memory_store':
        plan_steps = [
            PlanStep(
                step_description="Store user information in memory",
                module="memory",
                action="/store",
                parameters={"data": task_description}
            )
        ]

    elif intent == 'execution_task':
        plan_steps = [
            PlanStep(
                step_description="Execute a system-level command or operation",
                module="execution",
                action="/run_command",
                parameters={"command": task_description}
            )
        ]

    else:
        # Fallback step
        logger.warning("Intent not recognized, falling back to generic language processing.")
        plan_steps = [
            PlanStep(
                step_description="Process the task with a generic language model",
                module="language",
                action="/process_text",
                parameters={"text": task_description}
            )
        ]

    return plan_steps

# === API Endpoints ===

@app.post("/plan_task", response_model=PlanResponse)
async def plan_task(task_input: TaskInput):
    task_description = task_input.task_description
    logger.info(f"Received task: {task_description}")

    try:
        plan_steps = generate_plan(task_description)

        return PlanResponse(
            plan=plan_steps,
            status="success",
            message="Plan generated successfully"
        )
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating plan: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Reasoning service is up and running."}