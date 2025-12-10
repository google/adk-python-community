from google.adk.cli.service_registry import get_service_registry
from google.adk_community.sessions.redis_session_service import RedisSessionService


def redis_session_factory(uri: str, **kwargs):
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("agents_dir", None)
    return RedisSessionService(uri=uri, **kwargs_copy)


get_service_registry().register_session_service("redis", redis_session_factory)
