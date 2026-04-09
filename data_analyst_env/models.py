from openenv.core.env_server import Action, Observation, State


class SQLAction(Action):
    action_type: str  # 'query' or 'submit'
    query: str = ""
    answer: str = ""


class SQLObservation(Observation):
    output: str
    success: bool
    task_description: str
    reward: float
    is_done: bool


class SQLState(State):
    episode_id: str = ""
    step_count: int = 0
    task_level: str = "task_1"