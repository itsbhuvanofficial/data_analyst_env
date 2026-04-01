import sqlite3
import uuid
from typing import Tuple
from openenv.core.env_server import Environment
from models import SQLAction, SQLObservation, SQLState

class DataAnalystEnv(Environment):
    def __init__(self):
        super().__init__()
        # Explicitly initializing state to prevent Pydantic errors
        self._state = SQLState(episode_id="init", step_count=0, task_level="easy")
        self.conn = None
        self.tables_discovered = set()
        self.task_description = ""
        self.target_answer = ""
        
    def _setup_db(self):
        if self.conn:
            self.conn.close()
        self.conn = sqlite3.connect(":memory:")
        cursor = self.conn.cursor()
        
        # Build Schema & Seed Data
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, registered_year INTEGER)")
        cursor.execute("""
            INSERT INTO users VALUES 
            (1, 'Alice', 'alice@example.com', 2022),
            (2, 'Bob', 'bob@example.com', 2023),
            (3, 'Whale', 'whale@example.com', 2023),
            (4, 'Dave', 'dave@example.com', 2022),
            (5, 'Eve', 'eve@example.com', 2023)
        """)
        
        cursor.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price INTEGER)")
        cursor.execute("""
            INSERT INTO products VALUES 
            (1, 'Laptop', 'Electronics', 1000),
            (2, 'Mouse', 'Electronics', 50),
            (3, 'Desk', 'Furniture', 300),
            (4, 'Monitor', 'Electronics', 450)
        """)
        
        cursor.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER, amount INTEGER)")
        cursor.execute("""
            INSERT INTO orders VALUES 
            (1, 1, 1, 1000),
            (2, 2, 2, 50),
            (3, 3, 1, 2000),
            (4, 3, 4, 900),
            (5, 4, 3, 300)
        """)
        self.conn.commit()

    def reset(self, task_level: str = "easy") -> SQLObservation:
        self._state = SQLState(episode_id=str(uuid.uuid4()), task_level=task_level, step_count=0)
        self._setup_db()
        self.tables_discovered = set()
        
        # 3 Task levels evaluated by an exact programmatic grader (0.0 - 1.0 main score)
        tasks = {
            "easy": ("Count the total number of users in the database.", "5"),
            "medium": ("Find the email of the user who made the largest single order by amount.", "whale@example.com"),
            "hard": ("Calculate the total revenue from 'Electronics' products ordered by users registered in 2023. Return only the total amount as a number.", "2950") 
        }
        
        desc, target = tasks.get(task_level, tasks["easy"])
        self.task_description = desc
        self.target_answer = target
        
        # Pydantic requires explicit keyword arguments (e.g., output=...)
        return SQLObservation(
            output="Database connected. Hint: query sqlite_master to find tables.",
            success=True,
            task_description=self.task_description,
            reward=0.0,
            is_done=False
        )
        
    def step(self, action: SQLAction) -> SQLObservation:
        self._state.step_count += 1
        reward = 0.0
        is_done = False
        output = ""
        success = False

        # SAFETY CHECK: Ensure the user clicked /reset first!
        if self.conn is None:
            return SQLObservation(
                output="Error: Database not initialized. You MUST call /reset before calling /step.",
                success=False,
                task_description=self.task_description,
                reward=0.0,
                is_done=True
            )

        if action.action_type == "query":
            try:
                cursor = self.conn.cursor()
                cursor.execute(action.query)
                rows = cursor.fetchall()
                # Limit output to prevent massive LLM context overflows
                output = str(rows[:50]) + ("\n... (truncated)" if len(rows) > 50 else "")
                success = True
                
                # Meaningful Partial Progress Signals (Exploration Rewards)
                query_lower = action.query.lower()
                for table in["schema", "users", "products", "orders"]:
                    if (table in query_lower or (table=="schema" and "sqlite_master" in query_lower)) and table not in self.tables_discovered:
                        reward += 0.1
                        self.tables_discovered.add(table)
                        
            except Exception as e:
                output = f"SQL Error: {str(e)}"
                reward -= 0.02  # Minor penalty for invalid queries to encourage proper syntax
                
        elif action.action_type == "submit":
            is_done = True
            # Programmatic Grader Execution
            if action.answer.strip().lower() == self.target_answer.lower():
                output = "Correct! Task completed successfully."
                success = True
                reward += 1.0  # Main success reward
            else:
                output = f"Incorrect. Expected '{self.target_answer}', got '{action.answer}'"
                success = False
                reward += 0.0
        else:
            output = "Invalid action_type. Must be 'query' or 'submit'."
            reward -= 0.05

        # Terminate if too many steps
        if self._state.step_count >= 20 and not is_done:
            is_done = True
            output += "\nMax steps (20) reached. Episode terminated."

        # Pydantic requires explicit keyword arguments
        return SQLObservation(
            output=output, 
            success=success, 
            task_description=self.task_description, 
            reward=reward, 
            is_done=is_done
        )

    @property
    def state(self) -> SQLState:
        return self._state