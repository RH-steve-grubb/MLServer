# Serving ONNX models

Out of the box, `mlserver` supports the deployment and serving of `onnx` models.
By default, it will assume that these models have been [serialised as ONNX format files](https://onnx.ai/onnx/intro/converters.html).

In this example, we will cover how we can create and serialise a simple ONNX model, to then serve it using `mlserver`.


## Training

The first step will be to train a simple model using PyTorch and then convert it to ONNX format.
For that, we will use a simple linear regression model trained on synthetic data.



```python
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)


# Define a simple linear model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# Create and train the model
model = LinearRegressionModel(4, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print(f"Final training loss: {loss.item():.4f}")
```

### Saving our trained model

To save our trained model in ONNX format, we will use PyTorch's ONNX export functionality.
This is the [recommended approach by the ONNX project](https://onnx.ai/onnx/intro/converters.html).

Our model will be persisted as a file named `linear-model.onnx`.



```python
import torch.onnx

# Set the model to evaluation mode
model.eval()

# Create a dummy input for export (batch_size=1, features=4)
dummy_input = torch.randn(1, 4)

# Export the model to ONNX format
model_file_name = "linear-model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    model_file_name,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model saved as {model_file_name}")
```

## Serving

Now that we have trained and saved our model, the next step will be to serve it using `mlserver`. 
For that, we will need to create 2 configuration files: 

- `settings.json`: holds the configuration of our server (e.g. ports, log level, etc.).
- `model-settings.json`: holds the configuration of our model (e.g. input type, runtime to use, etc.).


##### `settings.json`



```python
%%writefile settings.json
{
    "debug": "true"
}

```

##### `model-settings.json`



```python
%%writefile model-settings.json
{
    "name": "linear-model",
    "implementation": "mlserver_onnx.OnnxModel",
    "parameters": {
        "uri": "./linear-model.onnx",
        "version": "v0.1.0"
    }
}

```

### Start serving our model

Now that we have our config in-place, we can start the server by running `mlserver start .`. This needs to either be ran from the same directory where our config files are or pointing to the folder where they are.

```shell
mlserver start .
```

Since this command will start the server and block the terminal, waiting for requests, this will need to be ran in the background on a separate terminal.


### Send test inference request

We now have our model being served by `mlserver`.
To make sure that everything is working as expected, let's send a request from our test set.

The request input `name` must match the ONNX model's input name (we used `"input"` in the export above). For that, we can use the Python types that `mlserver` provides out of box, or we can build our request manually.



```python
import requests

x_0 = X_test[0:1]
inference_request = {
    "inputs": [
        {"name": "input", "shape": x_0.shape, "datatype": "FP32", "data": x_0.tolist()}
    ]
}

endpoint = "http://localhost:8080/v2/models/linear-model/versions/v0.1.0/infer"
response = requests.post(endpoint, json=inference_request, timeout=30)

response.json()
```

As we can see above, the model predicted a value for the input, which we can compare with the actual test value.



```python
print(f"Predicted: {response.json()['outputs'][0]['data'][0]:.4f}")
print(f"Actual: {y_test[0]:.4f}")
```

### Requesting specific outputs

ONNX models expose named outputs. By default, MLServer returns the first output
using the `predict` alias. To request a specific output, define it explicitly.

Example request:



```python
import requests

x_0 = X_test[0:1]
inference_request = {
    "inputs": [
        {"name": "input", "shape": x_0.shape, "datatype": "FP32", "data": x_0.tolist()}
    ],
    "outputs": [{"name": "output"}],
}

endpoint = "http://localhost:8080/v2/models/linear-model/versions/v0.1.0/infer"
response = requests.post(endpoint, json=inference_request, timeout=30)

response.json()
```

### Execution providers and session options

If you need to tune threading or runtime options, configure ONNX Runtime via
`parameters.extra`.

Example configuration:



```python
%%writefile model-settings.json
{
  "name": "linear-model",
  "implementation": "mlserver_onnx.OnnxModel",
  "parameters": {
    "uri": "./linear-model.onnx",
    "version": "v0.1.0",
    "extra": {
      "providers": ["CPUExecutionProvider"],
      "provider_options": [{}],
      "session_options": {
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1
      },
      "session_config_entries": {
        "session.set_denormal_as_zero": "1"
      },
      "run_options": {
        "log_severity_level": 3
      }
    }
  }
}

```
