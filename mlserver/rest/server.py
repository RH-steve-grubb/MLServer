import uvicorn
import signal

from ..settings import Settings
from ..handlers import DataPlane, ModelRepositoryHandlers, get_custom_handlers
from ..model import MLModel

from .utils import matches
from .app import create_app
from .logging import (
    logger,
    disable_health_access_logs,
    clear_uvicorn_access_log_filters,
)


class _NoSignalServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass


class RESTServer:
    def __init__(
        self,
        settings: Settings,
        data_plane: DataPlane,
        model_repository_handlers: ModelRepositoryHandlers,
    ):
        self._settings = settings
        self._data_plane = data_plane
        self._model_repository_handlers = model_repository_handlers
        self._app = create_app(
            self._settings,
            data_plane=self._data_plane,
            model_repository_handlers=self._model_repository_handlers,
        )

    async def add_custom_handlers(self, model: MLModel) -> MLModel:
        handlers = get_custom_handlers(model)
        if not handlers:
            return model
        try:
            for custom_handler, handler_method in handlers:
                self._app.add_api_route(
                    custom_handler.rest_path,
                    handler_method,
                    methods=[custom_handler.rest_method],
                )
        except Exception:
            self._delete_custom_handlers(model)
            raise

        previous = model._on_load_failed

        def discard() -> None:
            # Routes call the instance directly, bypassing registry lookup.
            # Revoke them before waiting for abandoned native preparation;
            # retaining resource ownership must not retain serving eligibility.
            try:
                self._delete_custom_handlers(model)
            finally:
                previous()

        model._on_load_failed = discard

        return model

    async def delete_custom_handlers(self, model: MLModel) -> MLModel:
        self._delete_custom_handlers(model)
        return model

    def _delete_custom_handlers(self, model: MLModel) -> None:
        handlers = get_custom_handlers(model)
        if len(handlers) == 0:
            return

        # NOTE: Loop in reverse, so that it's quicker to find all the recently
        # added routes and we can remove routes on-the-fly
        for i, route in reversed(list(enumerate(self._app.routes))):
            for j, (custom_handler, handler_method) in enumerate(handlers):
                if matches(route, custom_handler, handler_method):  # type: ignore
                    self._app.routes.pop(i)
                    handlers.pop(j)

    async def start(self):
        cfg = self._get_config()
        self._server = _NoSignalServer(cfg)
        if self._settings.access_log and not self._settings.debug:
            disable_health_access_logs()

        logger.info(
            "HTTP server running on "
            f"http://{self._settings.host}:{self._settings.http_port}"
        )

        await self._server.serve()

    def _get_config(self):
        kwargs = {}

        if self._settings._custom_rest_server_settings:
            logger.warning(
                "REST custom configuration is out of support. Use as your own risk"
            )
            kwargs.update(self._settings._custom_rest_server_settings)

        kwargs.update(
            {
                "host": self._settings.host,
                "port": self._settings.http_port,
                "root_path": self._settings.root_path,
                "access_log": self._settings.access_log,
            }
        )

        if self._settings.logging_settings:
            # If not None, use ours. Otherwise, let Uvicorn fall back on its
            # own config.
            kwargs.update({"log_config": self._settings.logging_settings})

        return uvicorn.Config(self._app, **kwargs)

    async def stop(self, sig: int | None = None):
        if sig is None:
            # `sig` is no longer optional for `handle_exit` in
            # latest `uvicorn`
            sig = signal.SIGINT

        self._server.handle_exit(sig=sig, frame=None)
        # handle_exit returns immediately; drain runs in serve().
        clear_uvicorn_access_log_filters()
