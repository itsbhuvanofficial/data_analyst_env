from openenv.core.env_server import Action, Observation, State

class SQLAction(Action):
    """Action taken by the Data Analyst agent."""
    action_type: str  # 'query' or 'submit'
    query: str = ""   # SQL query to execute
    answer: str = ""  # Final answer to submit

class SQLObservation(Observation):
    """Observation returned to the agent."""
    output: str
    success: bool
    task_description: str
    reward: float
    is_done: bool

class SQLState(State):
    """Episode metadata and tracking."""
    episode_id: str = ""
    step_count: int = 0
    task_level: str = "easy"