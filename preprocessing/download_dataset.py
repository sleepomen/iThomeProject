import os
from dotenv import load_dotenv
from roboflow import Roboflow

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)

workspace_id = "ian-liu-s-workspace"
project_id = "find-airport"

project = rf.workspace(workspace_id).project(project_id)
version = project.version(1)
dataset = version.download("yolov8")

print("Finish")