from fastapi import status


class MLServerError(Exception):
    def __init__(self, msg: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(msg)
        self.status_code = status_code


class InvalidModelURI(MLServerError):
    def __init__(self, name: str, model_uri: str | None = None):
        msg = f"Invalid URI specified for model {name}"
        if model_uri:
            msg += f" ({model_uri})"

        super().__init__(msg, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ModelNotFound(MLServerError):
    def __init__(self, name: str, version: str | None = None):
        msg = f"Model {name} not found"
        if version is not None:
            msg = f"Model {name} with version {version} not found"

        super().__init__(msg, status.HTTP_404_NOT_FOUND)


class ModelNotReady(MLServerError):
    def __init__(self, name: str, version: str | None = None):
        msg = f"Model {name} is not ready yet."
        if version is not None:
            msg = f"Model {name} with version {version} is not ready yet."

        super().__init__(msg, status.HTTP_400_BAD_REQUEST)


class ModelLoadError(MLServerError):
    """Raised when model loading or validation fails (e.g. invalid artifact)."""

    cleanup_error: BaseException | None = None

    def __init__(self, msg: str):
        super().__init__(msg, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ModelUnloadError(MLServerError):
    """Raised when model unloading fails."""

    def __init__(self, msg: str):
        super().__init__(msg, status.HTTP_500_INTERNAL_SERVER_ERROR)


class ModelValidationError(MLServerError):
    """Raised when model configuration or options are invalid (e.g. at load time)."""

    def __init__(self, msg: str):
        super().__init__(msg, status.HTTP_422_UNPROCESSABLE_ENTITY)


class InferenceError(MLServerError):
    def __init__(self, msg: str):
        super().__init__(msg, status.HTTP_400_BAD_REQUEST)


class ModelParametersMissing(MLServerError):
    def __init__(self, model_name: str):
        super().__init__(
            f"Parameters missing for model {model_name}", status.HTTP_400_BAD_REQUEST
        )
