import mlflow
import pandas as pd
import numpy as np

# TODO: Set tht MLFlow server uri
uri = "http://127.0.0.1:6001/"
mlflow.set_tracking_uri(uri=uri)

# TODO: Provide model path/url
logged_model = "models:/cf55d88/latest"

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
np.random.seed(42)
data = np.random.rand(1, 64)

# TODO: Predict on a Pandas DataFrame. Due to the MLFlow functionality constrain.
#       The loaded model's predict function only accept dataframe as input instead of numpy array.
prediction=loaded_model.predict(pd.DataFrame(data))

# Print out prediction result
print(prediction)