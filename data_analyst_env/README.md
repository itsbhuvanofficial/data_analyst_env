# 📊 DataAnalystEnv (OpenEnv)

A real-world OpenEnv environment where an AI agent acts as a Database Administrator/Data Analyst. The agent is dropped into an unfamiliar SQL database and must autonomously explore the schema, write analytical queries, recover from syntax errors, and extract business intelligence.

This environment evaluates an LLM's capacity for **Text-to-SQL reasoning**, **tool use**, and **multi-step error recovery**—highly practical skills for enterprise AI agents.

---

## 🎯 Task Difficulty Levels (3 Tiers)
The environment features 3 distinct, deterministically graded tasks to evaluate frontier models:

1. **Easy:** "Count the total number of users in the database." *(Tests basic table querying).*
2. **Medium:** "Find the email of the user who made the largest single order by amount." *(Tests filtering, sorting, and single JOINs).*
3. **Hard:** "Calculate the total revenue from 'Electronics' products ordered by users registered in 2023." *(Tests complex multi-table JOINs, date filtering, and mathematical aggregation).*

---

## 🧩 Action & Observation Spaces

### Action Space (`SQLAction`)
The agent interacts with the environment by passing strict Pydantic-validated JSON:
* `action_type` (str): Must be either `"query"` (to execute SQL) or `"submit"` (to provide the final answer).
* `query` (str): The raw SQLite query to execute.
* `answer` (str): The final extracted metric to be evaluated by the grader.

### Observation Space (`SQLObservation`)
* `output` (str): The raw database execution return, limited to 50 rows to prevent LLM context-window overflow. If a syntax error occurs, the exact SQL engine traceback is returned here.
* `success` (bool): True if the query executed without errors.
* `reward` (float): The current reward signal.
* `is_done` (bool): Triggers `True` upon successful submission or hitting the 20-step limit.

---

## 🏆 Dense Reward Shaping
Unlike sparse environments that only reward `1.0` at the very end, `DataAnalystEnv` features dynamic, fractional reward shaping to accurately track partial AI progress (Score Variance):
* **+0.1 Exploration Reward:** Granted the first time an agent discovers a new, relevant table (`users`, `orders`, `products`, or querying `sqlite_master`).
* **-0.02 Syntax Penalty:** Applied when the agent writes invalid SQL, encouraging self-correction and clean code.
* **+1.0 Completion Reward:** Granted when the final programmatic exact-match grader verifies the submitted `answer`.

---

## 🚀 Local Setup & Testing

### 1. Run the Baseline Agent
Test the environment locally using the hardcoded baseline agent to verify determinism and API functionality:
```bash
pip install openenv-core requests
python run_inference.py