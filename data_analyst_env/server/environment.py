import sqlite3
import uuid
from typing import Tuple
from openenv.core.env_server import Environment
from models import SQLAction, SQLObservation, SQLState

class DataAnalystEnv(Environment):
    def __init__(self):
        super().__init__()
        # Initialize state with a default task
        self._state = SQLState(episode_id="init", step_count=0, task_level="task_1")
        self.conn = None
        self.tables_discovered = set()
        self.task_description = ""
        self.target_answer = ""
        
    def _setup_db(self):
        """Sets up the SQLite database in memory with sample data."""
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

    def reset(self, task_level: str = "task_1") -> SQLObservation:
        """Resets the environment for a new episode."""
        self._state = SQLState(episode_id=str(uuid.uuid4()), task_level=task_level, step_count=0)
        self._setup_db()
        self.tables_discovered = set()
        
        # 3 Task levels as required by Phase 2
        tasks = {
            "task_1": ("Count the total number of users in the database.", "5"),
            "task_2": ("Find the email of the user who made the largest single order by amount.", "whale@example.com"),
            "task_3": ("Calculate the total revenue from 'Electronics' products ordered by users registered in 2023. Return only the total amount as a number.", "2950") 
        }
        
        # Fallback to task_1 if level is not recognized
        desc, target = tasks.get(task_level, tasks["task_1"])
        self.task_description = desc
        self.target_answer = target
        
        # SCALING FIX: Start with a base reward of 0.05 
        # This ensures the score is NEVER exactly 0.0
        return SQLObservation(
            output=f"Task: {desc}\nDatabase connected. Query sqlite_master to start.",
            success=True,
            task_description=self.task_description,
            reward=0.05,
            is_done=False
        )
        
    def step(self, action: SQLAction) -> SQLObservation:
        """Executes one action and returns the observation and reward."""
        self._state.step_count += 1
        reward = 0.0
        is_done = False
        output = ""
        success = False

        if self.conn is None:
            return SQLObservation(
                output="Error: Database not initialized. Call /reset first.",
                success=False,
                task_description=self.task_description,
                reward=0.05,
                is_done=True
            )

        if action.action_type == "query":
            try:
                cursor = self.conn.cursor()
                cursor.execute(action.query)
                rows = cursor.fetchall()
                output = str(rows[:50]) + ("\n... (truncated)" if len(rows) > 50 else "")
                success = True
                
                # Tiny incremental rewards for discovery (max 0.03 total)
                query_lower = action.query.lower()
                for table in ["users", "products", "orders"]:
                    if table in query_lower and table not in self.tables_discovered:
                        reward += 0.01
                        self.tables_discovered.add(table)
                        
            except Exception as e:
                output = f"SQL Error: {str(e)}"
                reward -= 0.01 # Small penalty for syntax errors
                
        elif action.action_type == "submit":
            is_done = True
            # Programmatic Grader
            if action.answer.strip().lower() == self.target_answer.lower():
                output = "Correct! Task completed."
                success = True
                # SUCCESS REWARD: 0.85
                # Total possible: 0.05 (base) + 0.03 (discovery) + 0.85 (correct) = 0.93
                # This ensures we stay below 1.0 as required.
                reward = 0.85
            else:
                output = f"Incorrect submission."
                success = False
                reward = 0.0
        else:
            output = "Invalid action_type."
            reward = -0.01

        # Hard step limit
        if self._state.step_count >= 15 and not is_done:
            is_done = True
            output += "\nMax steps reached."

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