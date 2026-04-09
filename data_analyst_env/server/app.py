from openenv.core.env_server import create_fastapi_app
from models import SQLAction, SQLObservation
from .environment import DataAnalystEnv

shared_env = DataAnalystEnv()


def env_factory():
    return shared_env


app = create_fastapi_app(env_factory, SQLAction, SQLObservation)


@app.get("/")
def home():
    return {"status": "online"}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()