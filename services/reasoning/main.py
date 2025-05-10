# akshu-ai/services/reasoning/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class TaskInput(BaseModel):
    task_description: str
    context: Dict[str, Any] = {} # Relevant context from other modules (e.g., memory, user info)
    user_id: str = "anonymous"

class PlanStep(BaseModel):
    step_description: str
    module: str # The module responsible for this step (e.g., "language", "execution", "memory")
    action: str # The specific action/endpoint to call in the module
    parameters: Dict[str, Any] = {} # Parameters for the action

class PlanResponse(BaseModel):
    plan: List[PlanStep]
    status: str = "success"
    message: str = "Plan generated successfully"

@app.get("/")
def read_root():
    return {"message": "Reasoning service is running"}

@app.post("/plan_task", response_model=PlanResponse)
async def plan_task(task: TaskInput):
    print(f"Received task for planning from user {task.user_id}: {task.task_description}")

    # TODO: Implement actual reasoning and planning logic here
    # - Analyze the task description and context
    # - Break down the task into sub-steps
    # - Determine which modules/actions are needed for each step
    # - Consider constraints, user preferences (from persona module), and available tools
    # - This might involve rule-based systems, or even using an LLM for planning (as mentioned in docs)

    # --- Placeholder Planning Logic based on Keywords ---
    generated_plan: List[PlanStep] = []
    task_lower = task.task_description.lower()

    if "find information" in task_lower or "search" in task_lower:
        generated_plan.append(PlanStep(
            step_description="Search memory for relevant information.",
            module="memory",
            action="/retrieve",
            parameters={"query": task.task_description, "k": 5, "query_type": "semantic"}
        ))
        generated_plan.append(PlanStep(
            step_description="Process retrieved information using the language module.",
            module="language",
            action="/process", # Assuming language module can process retrieved docs
            parameters={"text": "Process retrieved memory data"} # TODO: Pass actual data
        ))

    elif "execute code" in task_lower or "run script" in task_lower:
         generated_plan.append(PlanStep(
            step_description="Prepare code for execution.",
            module="language",
            action="/process", # Potentially use language module to extract code
            parameters={"text": f"Extract code from: {task.task_description}"}
        ))
         generated_plan.append(PlanStep(
            step_description="Execute the extracted code.",
            module="execution",
            action="/execute_code",
            parameters={"code": "extracted_code_placeholder", "language": "python"} # TODO: Pass actual extracted code
        ))

    elif "analyze image" in task_lower or "what is this picture" in task_lower:
         generated_plan.append(PlanStep(
            step_description="Process the image using the vision module.",
            module="vision",
            action="/process_image",
            parameters={"image_data": "image_data_placeholder"} # TODO: Pass actual image data
        ))
         generated_plan.append(PlanStep(
            step_description="Describe the image analysis results using the language module.",
            module="language",
            action="/process",
            parameters={"text": "Describe vision analysis results"} # TODO: Pass actual results
        ))

    else:
        # Default plan: just use the language module
        generated_plan.append(PlanStep(
            step_description="Process the input using the language module.",
            module="language",
            action="/process",
            parameters={"text": task.task_description}
        ))

    print(f"Generated plan: {generated_plan}")

    return PlanResponse(plan=generated_plan)

# TODO: Add endpoints for evaluating plans, monitoring execution, etc.
