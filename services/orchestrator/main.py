import os
import json
import logging
import requests
import redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Dict, Any, List
from dotenv import load_dotenv
from prometheus_client import Counter, generate_latest
from starlette.responses import PlainTextResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AkshuAI Orchestrator",
    description="Central orchestrator for AkshuAI, coordinating tasks across language, memory, reasoning, and other modules.",
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
requests_counter = Counter("orchestrator_requests_total", "Total requests to orchestrator", ["endpoint"])

# Service URLs from environment variables
SERVICE_URLS = {
    "language": os.getenv("LANGUAGE_SERVICE_URL", "http://localhost:8001"),
    "memory": os.getenv("MEMORY_SERVICE_URL", "http://localhost:8002"),
    "vision": os.getenv("VISION_SERVICE_URL", "http://localhost:8004"),
    "execution": os.getenv("EXECUTION_SERVICE_URL", "http://localhost:8007"),
    "reasoning": os.getenv("REASONING_SERVICE_URL", "http://localhost:8006"),
    "persona": os.getenv("PERSONA_SERVICE_URL", "http://localhost:8003"),
    "audio": os.getenv("AUDIO_SERVICE_URL", "http://localhost:8005"),
}

# Timeouts configuration
TIMEOUTS = {
    "reasoning": int(os.getenv("REASONING_TIMEOUT", 10)),
    "language": int(os.getenv("LANGUAGE_TIMEOUT", 30)),
    "default": int(os.getenv("DEFAULT_SERVICE_TIMEOUT", 30))
}

# Redis client
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

# Pydantic models
class UserInput(BaseModel):
    text: str
    user_id: str = "anonymous"
    session_id: str = "default"
    context: Dict[str, Any] = {}

class PlanStep(BaseModel):
    step_description: str
    module: str
    action: str
    parameters: Dict[str, Any] = {}
    http_method: str = "POST"
    critical: bool = False
    output_key: str = ""  # New field to specify context key for output

class PlanResponse(BaseModel):
    plan: List[PlanStep]
    status: str = "success"
    message: str = "Plan generated successfully"

def resolve_parameters(parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    resolved = {}
    for key, value in parameters.items():
        if isinstance(value, str) and value.startswith("<PLACEHOLDER_") and value.endswith(">"):
            placeholder_key = value[11:-1].lower()
            resolved[key] = context.get(placeholder_key, value)
            logger.info(f"Resolved parameter '{key}' to '{resolved[key]}' using context key '{placeholder_key}'")
        else:
            resolved[key] = value
    return resolved

def synthesize_response(execution_results: List[Dict[str, Any]], language_service_url: str) -> str:
    """Synthesize a natural-language response from execution results, handling batch responses."""
    try:
        summary_prompt = "Summarize the following execution results in natural language:
"
        for result in execution_results:
            summary_prompt += f"- Step {result['step_number']}: {result['step_description']} ({result['status']})
"
            if result["result"]:
                if result["module"] == "language" and "generated_texts" in result["result"]:
                    for gen_text in result["result"]["generated_texts"]:
                        summary_prompt += f"  Result: {gen_text['generated_text']}
"
                elif result["module"] == "language" and "summary" in result["result"]:
                    summary_prompt += f"  Summary: {result['result']['summary']}
"
                elif result["module"] == "language" and "embeddings" in result["result"]:
                    summary_prompt += f"  Embeddings generated for {len(result['result']['embeddings'])} texts.
"
                elif result["module"] == "language" and "tokens" in result["result"]:
                    summary_prompt += f"  Tokenized {len(result['result']['tokens'])} texts.
"
                else:
                    summary_prompt += f"  Result: {json.dumps(result['result'], indent=2)}
"
            if result["error"]:
                summary_prompt += f"  Error: {result['error']}
"
        
        # Call the language module to generate the summary using the /generate endpoint
        # We assume the /generate endpoint can handle a single prompt for this summarization task
        response = requests.post(
            f"{language_service_url}/generate",
            json={
                "prompts": [summary_prompt], # Pass as a list for the /generate endpoint
                "max_tokens": 200 # Limit summary length
                # TODO: Pass user_id, session_id, and context if needed for persona/history in synthesis
            },
            timeout=TIMEOUTS["language"]
        )
        response.raise_for_status()
        # The /generate endpoint now returns 'text' key for orchestrator compatibility
        return response.json().get("text", "Failed to generate summary.")
    except Exception as e:
        logger.error(f"Error synthesizing response: {e}")
        # Fallback to a basic summary if language service call fails
        return "
".join([f"- Step {r['step_number']}: {r['step_description']} ({r['status']})" for r in execution_results])

async def fetch_persona(user_id: str, session_id: str) -> Dict[str, Any]:
    cache_key = f"persona:{user_id}:{session_id}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode cached persona for user {user_id}")
            # In case of decoding error, delete the invalid cache entry and fetch fresh
            redis_client.delete(cache_key)
            pass # Proceed to fetch fresh data

    
    try:
        response = requests.get(
            f"{SERVICE_URLS['persona']}/get_persona",
            params={
                "user_id": user_id,
                "session_id": session_id # Pass session_id if the persona service uses it
            },
            timeout=5
        )
        response.raise_for_status()
        persona = response.json().get("persona", {})
        # Cache the fetched persona data
        redis_client.setex(cache_key, 3600, json.dumps(persona)) # Cache for 1 hour
        return persona
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch persona for user {user_id}: {e}")
        return {} # Return empty persona on failure
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching persona for user {user_id}: {e}")
        return {} # Return empty persona on unexpected errors

@app.get("/")
async def read_root():
    requests_counter.labels(endpoint="/").inc()
    return {"message": "Orchestrator service is running"}

@app.get("/health")
async def health_check():
    requests_counter.labels(endpoint="/health").inc()
    # TODO: Add checks for dependent services (e.g., Reasoning, Language) if critical
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    requests_counter.labels(endpoint="/metrics").inc()
    return PlainTextResponse(generate_latest())

@app.post("/process_input")
async def process_input(input: UserInput, token: str = Depends(oauth2_scheme)):
    requests_counter.labels(endpoint="/process_input").inc()
    logger.info(f"Received input from user {input.user_id}: {input.text}")
    # Basic input sanitization (consider using a dedicated library like bleach)
    input.text = input.text.replace("<", "&lt;").replace(">", "&gt;")
    
    # --- Step 0: Fetch Persona (can be done in parallel with planning) ---
    # We fetch persona early as it might influence planning and execution.
    persona = await fetch_persona(input.user_id, input.session_id)
    orchestrated_response: Dict[str, Any] = {
        "input": input.dict(),
        "persona": persona, # Include persona in the response for visibility
        "plan": None,
        "execution_results": [],
        "final_response": ""
    }
    # Include persona in the context passed to the reasoning service
    reasoning_context = input.context.copy() # Start with user-provided context
    reasoning_context['persona'] = persona # Add fetched persona to context


    # --- Step 1: Get a plan from the Reasoning service ---
    try:
        reasoning_payload = {
            "task_description": input.text,
            "context": reasoning_context, # Pass the enriched context
            "user_id": input.user_id,
            "session_id": input.session_id # Pass session_id to reasoning if needed
        }
        logger.info(f"Calling Reasoning service with payload: {reasoning_payload}")
        # Use a timeout for the request
        reasoning_response = requests.post(
            f"{SERVICE_URLS['reasoning']}/plan_task",
            json=reasoning_payload,
            timeout=TIMEOUTS["reasoning"]
        )
        reasoning_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        plan_response_data = reasoning_response.json()
        # Validate the structure of the received plan data
        try:
            plan_response = PlanResponse(**plan_response_data)
            orchestrated_response["plan"] = plan_response.plan
            logger.info(f"Received valid plan from Reasoning service with {len(orchestrated_response['plan'])} steps.")
        except Exception as e:
             logger.error(f"Failed to parse plan response from Reasoning service: {e}")
             orchestrated_response["final_response"] = f"Invalid plan format from Reasoning service: {e}"
             return {"status": "error", "message": "Invalid plan format", "details": orchestrated_response}


        if not orchestrated_response["plan"]:
             orchestrated_response["final_response"] = "Reasoning service did not return a valid plan."
             logger.warning("Reasoning service did not return a valid plan.")
             return {"status": "error", "message": "Could not generate plan", "details": orchestrated_response}

    except requests.exceptions.Timeout:
        logger.error("Timeout when calling Reasoning service.")
        orchestrated_response["final_response"] = "Timeout when calling Reasoning service."
        return {"status": "error", "message": "Reasoning service timeout", "details": orchestrated_response}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Reasoning service: {e}")
        orchestrated_response["final_response"] = f"Error communicating with Reasoning service: {e}"
        return {"status": "error", "message": "Reasoning service error", "details": orchestrated_response}
    except Exception as e:
        logger.error(f"An unexpected error occurred during planning: {e}")
        orchestrated_response["final_response"] = f"An unexpected error occurred during planning: {e}"
        return {"status": "error", "message": "Unexpected planning error", "details": orchestrated_response}

    # --- Step 2: Execute the plan ---
    logger.info("Executing plan...")
    # Initialize current_context with the initial user context and fetched persona
    current_context: Dict[str, Any] = input.context.copy()
    current_context['persona'] = persona # Add persona to the execution context
    current_context['input_text'] = input.text # Make original input text available

    final_output_of_plan: Any = None

    for i, step_data in enumerate(orchestrated_response["plan"]):
        # Ensure the step data conforms to the PlanStep model
        try:
            step = PlanStep(**step_data)
        except Exception as e:
            logger.error(f"Invalid plan step format at index {i}: {e}")
            # Decide how to handle invalid steps - skip, log and continue, or halt?
            # For now, log and skip this step.
            step_result = {
                 "step_number": i + 1,
                 "step_description": "Invalid Step Format",
                 "module": "N/A",
                 "action": "N/A",
                 "status": "skipped",
                 "result": None,
                 "error": f"Invalid plan step format: {e}",
                 "parameters_sent": {}
            }
            orchestrated_response["execution_results"].append(step_result)
            continue # Skip to the next step

        step_result = {
            "step_number": i + 1,
            "step_description": step.step_description,
            "module": step.module,
            "action": step.action,
            "status": "failed",
            "result": None,
            "error": None,
            "parameters_sent": {}
        }
        logger.info(f"Executing step {step_result['step_number']}: {step.step_description} ({step.module}.{step.action})")

        try:
            module = step.module
            action = step.action
            parameters = step.parameters
            http_method = step.http_method.upper()
            output_key = step.output_key if step.output_key else step.step_description.lower().replace(" ", "_")

            if module not in SERVICE_URLS:
                step_result["error"] = f"Unknown module: {module}"
                logger.error(step_result["error"])
                orchestrated_response["execution_results"].append(step_result)
                # Decide whether to halt execution if a module is unknown - depends on criticality
                if step.critical:
                     logger.error(f"Critical step {step_result['step_number']} failed due to unknown module. Halting execution.")
                     orchestrated_response["final_response"] = f"Execution halted: Unknown module {module} in critical step."
                     return {"status": "failed", "message": "Critical step failed (Unknown module)", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}
                continue # Skip to the next step if module is unknown

            service_url = SERVICE_URLS[module]
            endpoint_url = f"{service_url}{action}"

            resolved_parameters = resolve_parameters(parameters, current_context)
            step_result["parameters_sent"] = resolved_parameters
            timeout = TIMEOUTS.get(module, TIMEOUTS["default"])

            logger.debug(f"Calling {module} service endpoint {action} with method {http_method} and parameters: {resolved_parameters}")
            
            response = None # Initialize response variable
            if http_method == "POST":
                response = requests.post(endpoint_url, json=resolved_parameters, timeout=timeout)
            elif http_method == "GET":
                response = requests.get(endpoint_url, params=resolved_parameters, timeout=timeout)
            # TODO: Add support for other HTTP methods (PUT, DELETE) if needed
            else:
                step_result["error"] = f"Unsupported HTTP method for step {step_result['step_number']}: {http_method}"
                logger.error(step_result["error"])
                orchestrated_response["execution_results"].append(step_result)
                if step.critical:
                     logger.error(f"Critical step {step_result['step_number']} failed due to unsupported HTTP method. Halting execution.")
                     orchestrated_response["final_response"] = f"Execution halted: Unsupported HTTP method {http_method} in critical step."
                     return {"status": "failed", "message": "Critical step failed (Unsupported HTTP method)", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}
                continue # Skip to the next step

            # Check if response object was successfully created
            if response is None:
                 step_result["error"] = f"Failed to create request object for step {step_result['step_number']}."
                 logger.error(step_result["error"])
                 orchestrated_response["execution_results"].append(step_result)
                 if step.critical:
                     logger.error(f"Critical step {step_result['step_number']} failed during request creation. Halting execution.")
                     orchestrated_response["final_response"] = f"Execution halted: Request creation failed in critical step."
                     return {"status": "failed", "message": "Critical step failed (Request creation)", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}
                 continue # Skip to the next step

            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            step_result["result"] = response.json()
            step_result["status"] = "succeeded"
            logger.info(f"Step {step_result['step_number']} succeeded.")

            # --- Update Context with Step Result ---
            # Use the specified output_key or a default based on step description
            # Store the result in the current context for subsequent steps
            current_context[output_key] = step_result["result"]
            logger.debug(f"Updated context with key '{output_key}': {current_context[output_key]}")

            # If this is the last step, or marked as the final output step, store its result
            # TODO: Add explicit way to mark a step as the final output step in the plan
            if i == len(orchestrated_response["plan"]) - 1:
                 final_output_of_plan = step_result["result"]
                 logger.info("This is the last step in the plan. Storing its result as final output.")

        except requests.exceptions.Timeout:
            step_result["error"] = f"Timeout when calling {module} service endpoint {action}."
            logger.error(f"Step {step_result['step_number']}: {step_result['error']}")
            # Decide how to handle timeouts - stop plan execution or continue?
            # For now, mark as failed and continue unless critical.
            if step.critical:
                 logger.error(f"Critical step {step_result['step_number']} timed out. Halting execution.")
                 orchestrated_response["final_response"] = f"Execution halted: Critical step timed out."
                 return {"status": "failed", "message": "Critical step timed out", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}
        except requests.exceptions.RequestException as e:
            step_result["error"] = f"Error calling {module} service endpoint {action}: {e}"
            logger.error(f"Step {step_result['step_number']}: {step_result['error']}")
            # Decide how to handle request errors - stop plan execution or continue?
            # For now, mark as failed and continue unless critical.
            if step.critical:
                 logger.error(f"Critical step {step_result['step_number']} failed with request error. Halting execution.")
                 orchestrated_response["final_response"] = f"Execution halted: Critical step request failed."
                 return {"status": "failed", "message": "Critical step request failed", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}
        except Exception as e:
            step_result["error"] = f"An unexpected error occurred during step execution: {e}"
            logger.error(f"Step {step_result['step_number']}: {e}")
            # Decide how to handle unexpected errors - stop plan execution or continue?
            # For now, mark as failed and continue unless critical.
            if step.critical:
                 logger.error(f"Critical step {step_result['step_number']} failed with unexpected error. Halting execution.")
                 orchestrated_response["final_response"] = f"Execution halted: Unexpected error in critical step."
                 return {"status": "failed", "message": "Critical step failed (Unexpected error)", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}

        orchestrated_response["execution_results"].append(step_result)
        # If a critical step failed and we haven't already returned, halt execution
        if step_result["status"] == "failed" and step.critical:
             # This case should ideally be caught by the specific exception handlers above,
             # but as a safeguard:
             if orchestrated_response["final_response"].startswith("Execution halted:"):
                 # Already handled and returning
                 return {"status": "failed", "message": "Execution halted", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}
             else:
                 # Should not happen with current error handling, but log as unexpected state
                 logger.error(f"Critical step {step_result['step_number']} failed, but execution was not explicitly halted by exception handler. Halting as safeguard.")
                 orchestrated_response["final_response"] = f"Execution halted due to critical step {step_result['step_number']} failure (safeguard). Error: {step_result['error']}"
                 return {"status": "failed", "message": "Critical step failed (Safeguard halt)", "details": orchestrated_response, "final_response": orchestrated_response["final_response"]}


    # --- Step 3: Synthesize final response ---
    logger.info("Synthesizing final response.")
    # The synthesis now uses the language module's /generate endpoint
    # Pass the execution results to the language module for summarization
    orchestrated_response["final_response"] = synthesize_response(
        orchestrated_response["execution_results"],
        SERVICE_URLS["language"]
    )

    # Determine overall status based on step execution
    overall_status = "succeeded"
    for result in orchestrated_response["execution_results"]:
        if result["status"] == "failed":
            overall_status = "partially_succeeded" # Or "failed" if any failure means overall failure
            # If any step was critical and failed, the status should already be 'failed' and we would have returned earlier.
            # If a non-critical step failed, overall status is 'partially_succeeded'.
            # If a critical step failed but we continued (e.g., unknown module not marked critical), this would be 'partially_succeeded' incorrectly.
            # Let's ensure critical failures lead to 'failed' status earlier.
            # With the updated critical failure handling, this logic for 'partially_succeeded' is for non-critical failures.

    # If the orchestrator itself failed to get a plan or had an early error, the status would reflect that.
    # If all steps were skipped or there were no steps, what should the status be?
    if not orchestrated_response["execution_results"] and orchestrated_response["plan"] is not None and len(orchestrated_response["plan"]) > 0:
         # This case implies all steps were skipped or somehow missed execution
         overall_status = "failed" # Or a specific 'execution_skipped' status
         orchestrated_response["final_response"] = orchestrated_response["final_response"] if orchestrated_response["final_response"] else "Plan execution skipped or failed unexpectedly."
         logger.error("Plan execution resulted in no steps or all steps skipped.")
    elif not orchestrated_response["execution_results"] and (orchestrated_response["plan"] is None or len(orchestrated_response["plan"]) == 0):
        # Case where no plan was generated or plan was empty - status should reflect the planning phase outcome
        # The initial planning step already sets the status and returns in this case.
        pass # Do nothing, status and response already set

    # Final return structure
    return {
        "status": overall_status,
        "message": "Input processed and plan executed",
        "details": orchestrated_response,
        "final_response": orchestrated_response["final_response"]
    }

# Add more endpoints here to handle routing and orchestration logic
# TODO: Add endpoints for managing sessions, users, system status, etc.
# TODO: Implement a more robust and explicit way to define the *final output* of a plan.
# Currently, it's implicitly the result of the last step, which might not always be desired.
