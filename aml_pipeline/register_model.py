# register_model.py
from azureml.core import Workspace, Model

# Connect to workspace
ws = Workspace.from_config()

# Register the model
model = Model.register(workspace=ws,
                       model_name="iris_model",
                       model_path="outputs/iris_model.joblib")
print("Model registered: ", model.name)
