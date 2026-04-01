from openenv.core.env_server import create_fastapi_app
from models import SQLAction, SQLObservation
from server.environment import DataAnalystEnv

# 1. Create a single, persistent instance of our environment
shared_env = DataAnalystEnv()

# 2. Create a "factory function" that always returns this exact same instance
def env_factory():
    return shared_env

# 3. Pass the factory function to FastAPI so it shares the memory!
app = create_fastapi_app(env_factory, SQLAction, SQLObservation)