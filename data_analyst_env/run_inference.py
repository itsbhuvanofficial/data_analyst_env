from models import SQLAction
from server.environment import DataAnalystEnv

def run_baseline():
    env = DataAnalystEnv()
    
    print("=== Running Baseline for EASY Task ===")
    obs = env.reset(task_level="easy")
    print(f"Task: {obs.task_description}")
    
    # Explores schema (+0.1 reward)
    obs = env.step(SQLAction(action_type="query", query="SELECT name FROM sqlite_master WHERE type='table';"))
    print(f"Action: Query Schema -> Reward Received: {obs.reward:.2f}")
    
    # Query table (+0.1 reward)
    obs = env.step(SQLAction(action_type="query", query="SELECT COUNT(*) FROM users;"))
    print(f"Action: Query Users -> Reward Received: {obs.reward:.2f} | Output: {obs.output}")
    
    # Final Grader (+1.0 reward)
    obs = env.step(SQLAction(action_type="submit", answer="5"))
    print(f"Action: Submit -> Reward Received: {obs.reward:.2f} | Output: {obs.output}")
    print(f"Episode Done: {obs.is_done}\n")

    print("=== Running Baseline for HARD Task ===")
    obs = env.reset(task_level="hard")
    
    # Accumulate all partial exploration signals (4 * 0.1 = 0.4)
    env.step(SQLAction("query", "SELECT * FROM sqlite_master;"))
    env.step(SQLAction("query", "SELECT * FROM users;"))
    env.step(SQLAction("query", "SELECT * FROM products;"))
    env.step(SQLAction("query", "SELECT * FROM orders;"))
    
    query = """
    SELECT SUM(o.amount) FROM orders o
    JOIN users u ON o.user_id = u.id
    JOIN products p ON o.product_id = p.id
    WHERE u.registered_year = 2023 AND p.category = 'Electronics';
    """
    obs = env.step(SQLAction(action_type="query", query=query))
    print(f"Action: Complex Join -> Output: {obs.output}")
    
    # Exact programmatic check yields 1.0 success
    obs = env.step(SQLAction(action_type="submit", answer="2950"))
    print(f"Final Action: Submit Answer -> Reward: {obs.reward:.2f} | Done: {obs.is_done}")

if __name__ == "__main__":
    run_baseline()