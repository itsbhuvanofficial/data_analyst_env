import sqlite3
import uuid
from openenv.core.env_server import Environment
from models import SQLAction, SQLObservation, SQLState


class DataAnalystEnv(Environment):
    def __init__(self):
        super().__init__()
        self._state = SQLState(episode_id="init", step_count=0)
        self.conn = None
        self.task_description = ""
        self.target_answer = ""
        self._explored_tables = set()

    def _setup_db(self):
        if self.conn:
            self.conn.close()
        self.conn = sqlite3.connect(":memory:")
        cursor = self.conn.cursor()

        cursor.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, registered_year INTEGER)"
        )
        cursor.execute(
            "INSERT INTO users VALUES "
            "(1, 'Alice', 'alice@example.com', 2022), "
            "(2, 'Bob', 'bob@example.com', 2023), "
            "(3, 'Whale', 'whale@example.com', 2023), "
            "(4, 'Dave', 'dave@example.com', 2022), "
            "(5, 'Eve', 'eve@example.com', 2023)"
        )
        cursor.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price INTEGER)"
        )
        cursor.execute(
            "INSERT INTO products VALUES "
            "(1, 'Laptop', 'Electronics', 1000), "
            "(2, 'Mouse', 'Electronics', 50), "
            "(3, 'Desk', 'Furniture', 300), "
            "(4, 'Monitor', 'Electronics', 450)"
        )
        cursor.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER, amount INTEGER)"
        )
        cursor.execute(
            "INSERT INTO orders VALUES "
            "(1, 1, 1, 1000), "
            "(2, 2, 2, 50), "
            "(3, 3, 1, 2000), "
            "(4, 3, 4, 900), "
            "(5, 4, 3, 300)"
        )
        self.conn.commit()

    def reset(self, task_id: str = "task_1", **kwargs) -> SQLObservation:
        t_id = task_id or kwargs.get("task_level", "task_1")

        self._state = SQLState(episode_id=str(uuid.uuid4()), task_level=t_id, step_count=0)
        self._explored_tables = set()
        self._setup_db()

        tasks = {
            "task_1": ("Count the total number of users in the database.", "5"),
            "task_2": (
                "Find the email of the user who made the largest single order by amount.",
                "whale@example.com",
            ),
            "task_3": (
                "Calculate the total revenue from 'Electronics' products ordered by users registered in 2023.",
                "2950",
            ),
        }

        desc, target = tasks.get(t_id, tasks["task_1"])
        self.task_description = desc
        self.target_answer = target

        # Base reward strictly between 0 and 1
        return SQLObservation(
            output=f"Environment ready. Task: {desc}",
            success=True,
            task_description=desc,
            reward=0.05,
            is_done=False,
        )

    def step(self, action: SQLAction) -> SQLObservation:
        self._state.step_count += 1
        reward = 0.0
        is_done = False
        output = ""
        success = False

        if action.action_type == "query":
            try:
                cursor = self.conn.cursor()
                cursor.execute(action.query)
                rows = cursor.fetchall()[:50]
                output = str(rows)
                success = True

                # Exploration bonus: first time a relevant table is touched
                query_lower = action.query.lower()
                for table in ["users", "orders", "products", "sqlite_master"]:
                    if table in query_lower and table not in self._explored_tables:
                        self._explored_tables.add(table)
                        reward += 0.1

                reward += 0.01  # small step reward
            except Exception as e:
                output = str(e)
                reward = -0.02  # syntax penalty

        elif action.action_type == "submit":
            is_done = True
            if action.answer.strip().lower() == self.target_answer.strip().lower():
                output = "Correct!"
                success = True
                reward = 0.85
            else:
                output = f"Incorrect. Expected: {self.target_answer}"
                reward = 0.0

        if self._state.step_count >= 20:
            is_done = True

        return SQLObservation(
            output=output,
            success=success,
            task_description=self.task_description,
            reward=reward,
            is_done=is_done,
        )

    @property
    def state(self) -> SQLState:
        return self._state