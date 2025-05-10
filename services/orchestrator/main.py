# akshu-ai/services/orchestrator/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import requests
import json # Import json for potential debugging

app = FastAPI()

# Assuming the services are running locally on different ports for this example
# In a real deployment, use service discovery (e.g., Kubernetes service names)
SERVICE_URLS = {
    "language": "http://localhost:8001",
    "memory": "http://localhost:8002",
    "vision": "http://localhost:8004",
    "execution": "http://localhost:8007",
    "reasoning": "http://localhost:8006", # Added Reasoning service URL
    "persona": "http://localhost:8003", # Added Persona service URL (although not used in process_input yet)
    "audio": "http://localhost:8005", # Added Audio service URL (although not used in process_input yet)
}

class UserInput(BaseModel):
    text: str
    user_id: str = "anonymous"
    session_id: str = "default"
    context: Dict[str, Any] = {} # Additional context for the reasoning module

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
    return {"message": "Orchestrator service is running"}

@app.post("/process_input")
async def process_input(input: UserInput):
    print(f"Received input from user {input.user_id}: {input.text}")

    orchestrated_response: Dict[str, Any] = {
        "input": input.dict(),
        "plan": None,
        "execution_results": [],
        "final_response": ""
    }

    # --- Step 1: Get a plan from the Reasoning service ---
    try:
        reasoning_payload = {
            "task_description": input.text,
            "context": input.context, # Pass any additional context
            "user_id": input.user_id
        }
        print(f"Calling Reasoning service with payload: {reasoning_payload}")
        reasoning_response = requests.post(f"{SERVICE_URLS['reasoning']}/plan_task", json=reasoning_payload)
        reasoning_response.raise_for_status()
        plan_response_data = reasoning_response.json()
        orchestrated_response["plan"] = plan_response_data.get("plan")
        print(f"Received plan from Reasoning service: {orchestrated_response['plan']}")

        if not orchestrated_response["plan"]:
             orchestrated_response["final_response"] = "Reasoning service did not return a valid plan."
             print("Reasoning service did not return a valid plan.")
             return {"status": "error", "message": "Could not generate plan", "details": orchestrated_response}

    except requests.exceptions.RequestException as e:
        print(f"Error calling Reasoning service: {e}")
        orchestrated_response["final_response"] = f"Error communicating with Reasoning service: {e}"
        return {"status": "error", "message": "Reasoning service error", "details": orchestrated_response}
    except Exception as e:
        print(f"An unexpected error occurred during planning: {e}")
        orchestrated_response["final_response"] = f"An unexpected error occurred during planning: {e}"
        return {"status": "error", "message": "Unexpected planning error", "details": orchestrated_response}

    # --- Step 2: Execute the plan ---
    print("Executing plan...")
    for step in orchestrated_response["plan"]:
        step_result = {
            "step_description": step.get("step_description"),
            "module": step.get("module"),
            "action": step.get("action"),
            "status": "failed",
            "result": None,
            "error": None
        }
        try:
            module = step.get("module")
            action = step.get("action")
            parameters = step.get("parameters", {})

            if module not in SERVICE_URLS:
                step_result["error"] = f"Unknown module: {module}"
                print(f"Unknown module specified in plan: {module}")
                orchestrated_response["execution_results"].append(step_result)
                continue # Skip to the next step if module is unknown

            service_url = SERVICE_URLS[module]
            endpoint_url = f"{service_url}{action}"

            print(f"Executing step: Calling {module} service endpoint {action} with parameters: {parameters}")

            # TODO: Handle different HTTP methods (GET, POST, etc.) based on the action/module
            # Currently assumes POST for all actions with JSON payload
            response = requests.post(endpoint_url, json=parameters)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            step_result["result"] = response.json()
            step_result["status"] = "succeeded"
            print(f"Step succeeded. Result: {step_result['result']}")

        except requests.exceptions.RequestException as e:
            step_result["error"] = f"Error calling {module} service: {e}"
            print(f"Error executing step - calling {module} service: {e}")
        except Exception as e:
            step_result["error"] = f"An unexpected error occurred during step execution: {e}"
            print(f"An unexpected error occurred during step execution: {e}")

        orchestrated_response["execution_results"appyend(step_result)

    # --- Step 3: Synthesize Final Response ---
    # TODO: Implement sophisticated response synthesis based on all execution_results
    # This might involve the language module to generate a natural language summary.

    final_response_parts = []
    if orchestrated_response["execution_results"]:
        final_response_parts.append("Execution Results:")
        for result in orchestrated_response["execution_results"]:
            final_response_parts.append(f"- Step '{result['step_description']}' ({result['module']}.{result['action']}): {result['status']}")
            if result['status'] == "succeeded":
                 # Attempt to pretty print successful results if they are dictionaries
                 try:
                     final_response_parts.append(f"  Result: {json.dumps(result['result'], indent=2)}")
                 except (TypeError, json.JSONDecodeError):
                      final_response_parts.append(f"  Result: {result['result']}")
            else:
                final_response_parts.append(f"  Error: {result['error']}")
    else:
        final_response_parts.append("No execution steps were performed.")

    orchestrated_response["final_response"] = "
".join(final_response_parts)

    return {"status": "success", "message": "Input processed and plan executed", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}

# Add more endpoints here to handle routing and orchestration logic
# TODO: Add endpoints for managing sessions, users, system status, etc.
