from openenv.core.env_server import create_fastapi_app
from models import SQLAction, SQLObservation
from .environment import DataAnalystEnv

shared_env = DataAnalystEnv()

def env_factory():
    return shared_env

# The FastAPI app object
app = create_fastapi_app(env_factory, SQLAction, SQLObservation)

@app.get("/")
def home():
    return {"status": "Environment is Live", "docs": "/docs"}

# Fix 1 & 2: Rename 'start' to 'main'
def main():
    import uvicorn
    # This must match your folder structure: server.app:app
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

# Fix 3: Add the mandatory execution block
if __name__ == "__main__":
    main()