import logging
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from API_servers.router import (
    high_throughput_router,
    openai_router,
    state_router,
    v1_router,
)
from infer.high_throughput import HighThroughputResidentRuntime


def create_app(engine, password=None, high_throughput_config=None):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.engine = engine
    app.state.password = password
    app.state.dialogue_idx_lock = Lock()
    app.state.dialogue_idx_counters = {}

    app.include_router(v1_router)
    if high_throughput_config is not None and high_throughput_config.enabled:
        app.state.high_throughput_runtime = HighThroughputResidentRuntime(
            engine,
            high_throughput_config,
        )
        app.include_router(high_throughput_router)
    app.include_router(state_router)
    app.include_router(openai_router)

    @app.on_event("startup")
    async def log_registered_routes():
        logger = logging.getLogger("uvicorn.error")
        logger.info("Registered FastAPI routes:")
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue
            methods = ",".join(sorted(route.methods - {"HEAD", "OPTIONS"}))
            logger.info("  %-20s %s", methods, route.path)

    return app
