import os
import json
from openai import OpenAI
from models import SQLAction
from server.environment import DataAnalystEnv

# ======================================================================
# STRICT PRE-SUBMISSION CHECKLIST COMPLIANCE
# ======================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ======================================================================
# MANDATORY STRUCTURED LOGGING
# ======================================================================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action} reward={reward:+.2f} done={done}{err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

# ======================================================================
# INFERENCE LOGIC
# ======================================================================
def get_model_action(client: OpenAI, task_desc: str, obs_output: str, history: list):
    """Uses the LLM to decide the next action based on the environment observation."""
    
    sys_prompt = """You are an expert Data Analyst AI agent connected to a SQLite database.
    Your goal is to complete the task by exploring the schema, writing queries, and submitting the final answer.
    
    You MUST respond with a raw JSON object matching this schema exactly:
    {
        "action_type": "query" or "submit",
        "query": "SQL query here if action_type is query, else empty",
        "answer": "final answer here if action_type is submit, else empty"
    }"""
    
    messages =[{"role": "system", "content": sys_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h})
        
    messages.append({
        "role": "user", 
        "content": f"Task: {task_desc}\nLatest Observation: {obs_output}\nWhat is your next action JSON?"
    })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"action_type": "error", "error": str(e)}

def main():
    # Only run the LLM loop if the user actually provided an API key. 
    # Otherwise, fallback gracefully so the automated tests don't crash.
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy-key")
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    
    env = DataAnalystEnv()
    
    # UPDATED: Use the new Task ID and adjusted Reward Cap for Phase 2
    TASK_LEVEL = "task_3" 
    MAX_STEPS = 10
    MAX_TOTAL_REWARD = 0.95 # Base(0.05) + Discovery(0.03) + Success(0.85) = 0.93 max
    
    history = []
    rewards =[]
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_LEVEL, env="DataAnalystEnv", model=MODEL_NAME)

    try:
        # Reset gives the initial 0.05 reward
        obs = env.reset(task_level=TASK_LEVEL)
        rewards.append(obs.reward)
        
        for step in range(1, MAX_STEPS + 1):
            if obs.is_done:
                break
                
            steps_taken = step
            
            # 1. Ask LLM for the next action
            action_dict = get_model_action(client, obs.task_description, obs.output, history)
            
            error = None
            if action_dict.get("action_type") == "error":
                error = action_dict.get("error")
                action = SQLAction(action_type="query", query="SELECT 1;") # dummy fallback action
            else:
                action = SQLAction(
                    action_type=action_dict.get("action_type", "query"),
                    query=action_dict.get("query", ""),
                    answer=action_dict.get("answer", "")
                )
            
            # 2. Take the step in the environment
            action_str = f"{action.action_type}: {action.query or action.answer}"
            obs = env.step(action)
            
            reward = obs.reward or 0.0
            done = obs.is_done
            
            rewards.append(reward)
            
            # 3. Log the step exactly as required
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            history.append(f"Action I took: {action_str}\nObservation received: {obs.output}")

            if done:
                break

        # Calculate final score (will be strictly between 0 and 1)
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.05
        score = min(max(score, 0.01), 0.99) # Extra safety for range boundaries
        success = score >= 0.7  # Passing threshold

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()