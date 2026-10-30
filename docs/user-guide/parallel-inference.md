# Parallel Inference

Out of the box, MLServer includes support to offload inference workloads to a
pool of workers running in separate processes.
This allows MLServer to scale out beyond the limitations of the Python
interpreter.
To learn more about why this can be beneficial, you can check the [concurrency
section](#concurrency-in-python) below.

![](../assets/parallel-inference.svg)

By default, MLServer will spin up a pool with only one worker process to run
inference.
All models will be loaded uniformly across the inference pool workers.
To read more about advanced settings, please see the [usage section
below](#usage).

## Concurrency in Python

The [Global Interpreter Lock
(GIL)](https://wiki.python.org/moin/GlobalInterpreterLock) is a mutex lock that
exists in most Python interpreters (e.g. CPython).
Its main purpose is to lock Python’s execution so that it only runs on a single
processor at the same time.
This simplifies certain things to the interpreter.
However, it also adds the limitation that a **single Python process will never
be able to leverage multiple cores**.

When we think about MLServer's support for [multi-model serving via the Model
Repository](../examples/model-repository/README.md), this could lead to scenarios where a
**heavily-used model starves the other models** running within the same
MLServer instance.
Similarly, even if we don't take multi-model serving into account, the **GIL also makes it harder
to scale inference for a single model**.

To work around this limitation, MLServer offloads the model inference to a pool
of workers, where each worker is a separate Python process (and thus has its
own separate GIL).
This means that we can get full access to the underlying hardware.

### Overhead

Managing the Inter-Process Communication (IPC) between the main MLServer
process and the inference pool workers brings in some overhead.
MLServer manages worker processes through `multiprocessing` and uses
`multiprocessing.Queue` for inference messages by default. The optional ZeroMQ
transport changes inference IPC while retaining queues for model control.
Both transports serialize requests and responses. See
[Inference Performance Options](./performance.md) for configuration and the
related REST and gRPC conversion options.

IPC overhead can be **particularly noticeable for lightweight inference
methods**, where it can take a large percentage of the overall time.
In these cases (which can only be assessed on a model-by-model basis), the user
has the option to [disable the parallel inference feature](#usage).

For regular models where inference can take a bit more time, this overhead is
usually offset by the benefit of having multiple cores to compute inference on.

## Usage

By default, MLServer will always create an inference pool with one single
worker.
The number of workers (i.e. the size of the inference pool) can be adjusted
globally through the server-level `parallel_workers` setting.

### `parallel_workers`

The `parallel_workers` field of the `settings.json` file (or alternatively, the
`MLSERVER_PARALLEL_WORKERS` global environment variable) controls the size of
MLServer's inference pool.
The expected values are:

- `N`, where `N > 0`, will create a pool of `N` workers.
- `0`, will disable the parallel inference feature.
  In other words, inference will happen within the main MLServer process.
