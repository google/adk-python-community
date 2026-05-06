# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import bisect
import logging
import time
import uuid
from typing import Any, Optional

import orjson
import redis.asyncio as redis
from redis.crc import key_slot
from typing_extensions import override

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State

from .utils import _json_serializer

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_EXPIRATION = 60 * 60  # 1 hour


def _session_serializer(obj: Session) -> bytes:
    """Serialize ADK Session to JSON bytes."""
    return orjson.dumps(obj.model_dump(), default=_json_serializer)


class RedisKeys:
    """Helper to generate Redis keys consistently."""

    @staticmethod
    def session(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def user_sessions(app_name: str, user_id: str) -> str:
        return f"{State.APP_PREFIX}:{app_name}:{user_id}"

    @staticmethod
    def app_state(app_name: str) -> str:
        return f"{State.APP_PREFIX}{app_name}"

    @staticmethod
    def user_state(app_name: str, user_id: str) -> str:
        return f"{State.USER_PREFIX}{app_name}:{user_id}"


class RedisSessionService(BaseSessionService):
    """A Redis-backed implementation of the session service."""

    def __init__(
        self,
        host="localhost",
        port=6379,
        db=0,
        uri=None,
        cluster_uri=None,
        expire=DEFAULT_EXPIRATION,
        **kwargs,
    ):
        self.expire = expire

        if cluster_uri:
            self.cache = redis.RedisCluster.from_url(cluster_uri, **kwargs)
        elif uri:
            self.cache = redis.Redis.from_url(uri, **kwargs)
        else:
            self.cache = redis.Redis(host=host, port=port, db=db, **kwargs)

    async def health_check(self) -> bool:
        try:
            await self.cache.ping()
            return True
        except redis.RedisError:
            return False

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        session_id = (
            session_id.strip()
            if session_id and session_id.strip()
            else str(uuid.uuid4())
        )
        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=state or {},
            last_update_time=time.time(),
        )

        user_sessions_key = RedisKeys.user_sessions(app_name, user_id)
        session_key = RedisKeys.session(session_id)

        async with self.cache.pipeline(transaction=False) as pipe:
            pipe.sadd(user_sessions_key, session_id)
            pipe.expire(user_sessions_key, self.expire)
            pipe.set(
                session_key,
                _session_serializer(session),
                ex=self.expire,
            )
            await pipe.execute()

        return await self._merge_state(app_name, user_id, session)

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        session_key = RedisKeys.session(session_id)
        raw_session = await self.cache.get(session_key)
        if not raw_session:
            user_sessions_key = RedisKeys.user_sessions(app_name, user_id)
            await self.cache.srem(user_sessions_key, session_id)
            return None

        try:
            session_dict = orjson.loads(raw_session)
            session = Session.model_validate(session_dict)
        except (orjson.JSONDecodeError, Exception) as e:
            logger.error(f"Error decoding session {session_id}: {e}")
            return None

        if config:
            if config.num_recent_events:
                session.events = session.events[-config.num_recent_events :]
            if config.after_timestamp:
                timestamps = [e.timestamp for e in session.events]
                start_index = bisect.bisect_left(timestamps, config.after_timestamp)
                session.events = session.events[start_index:]

        return await self._merge_state(app_name, user_id, session)

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: str
    ) -> ListSessionsResponse:
        sessions = await self._load_sessions(app_name, user_id)
        sessions_without_events = []

        for session_data in sessions.values():
            session = Session.model_validate(session_data)
            session.events = []
            session.state = {}
            sessions_without_events.append(session)

        return ListSessionsResponse(sessions=sessions_without_events)

    @override
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        user_sessions_key = RedisKeys.user_sessions(app_name, user_id)
        session_key = RedisKeys.session(session_id)

        async with self.cache.pipeline(transaction=False) as pipe:
            pipe.srem(user_sessions_key, session_id)
            pipe.delete(session_key)
            await pipe.execute()

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        await super().append_event(session=session, event=event)
        session.last_update_time = event.timestamp

        async with self.cache.pipeline(transaction=False) as pipe:
            user_sessions_key = RedisKeys.user_sessions(
                session.app_name, session.user_id
            )
            pipe.expire(user_sessions_key, self.expire)

            if event.actions and event.actions.state_delta:
                for key, value in event.actions.state_delta.items():
                    if key.startswith(State.APP_PREFIX):
                        pipe.hset(
                            RedisKeys.app_state(session.app_name),
                            key.removeprefix(State.APP_PREFIX),
                            orjson.dumps(value),
                        )
                    if key.startswith(State.USER_PREFIX):
                        pipe.hset(
                            RedisKeys.user_state(session.app_name, session.user_id),
                            key.removeprefix(State.USER_PREFIX),
                            orjson.dumps(value),
                        )

            pipe.set(
                RedisKeys.session(session.id),
                _session_serializer(session),
                ex=self.expire,
            )
            await pipe.execute()

        return event

    async def _merge_state(
        self, app_name: str, user_id: str, session: Session
    ) -> Session:
        app_state = await self.cache.hgetall(RedisKeys.app_state(app_name))
        for k, v in app_state.items():
            session.state[State.APP_PREFIX + k.decode()] = orjson.loads(v)

        user_state = await self.cache.hgetall(RedisKeys.user_state(app_name, user_id))
        for k, v in user_state.items():
            session.state[State.USER_PREFIX + k.decode()] = orjson.loads(v)

        return session

    async def _load_sessions(self, app_name: str, user_id: str) -> dict[str, dict]:
        key = RedisKeys.user_sessions(app_name, user_id)
        try:
            session_ids_bytes = await self.cache.smembers(key)
            if not session_ids_bytes:
                return {}

            session_ids = [s.decode() for s in session_ids_bytes]
            session_keys = [RedisKeys.session(sid) for sid in session_ids]

            # Group by slot for Redis Cluster
            slot_groups: dict[int, list[str]] = {}
            for k in session_keys:
                slot = key_slot(k.encode())
                slot_groups.setdefault(slot, []).append(k)

            async def fetch_group(keys: list[str]):
                async with self.cache.pipeline(transaction=False) as pipe:
                    for k in keys:
                        pipe.get(k)
                    return await pipe.execute()

            results_per_group = await asyncio.gather(
                *(fetch_group(keys) for keys in slot_groups.values())
            )

            raw_sessions = []
            for group_keys, group_results in zip(
                slot_groups.values(), results_per_group
            ):
                raw_sessions.extend(zip(group_keys, group_results))

            sessions = {}
            sessions_to_cleanup = []
            for key_name, raw_session in raw_sessions:
                session_id = key_name.split(":", 1)[1]
                if raw_session:
                    try:
                        sessions[session_id] = orjson.loads(raw_session)
                    except orjson.JSONDecodeError as e:
                        logger.error(f"Error decoding session {session_id}: {e}")
                else:
                    logger.warning(
                        "Session ID %s found in user set but session data is missing. Cleaning up.",
                        session_id,
                    )
                    sessions_to_cleanup.append(session_id)

            if sessions_to_cleanup:
                await self.cache.srem(key, *sessions_to_cleanup)

            return sessions
        except redis.RedisError as e:
            logger.error(f"Error loading sessions for {user_id}: {e}")
            return {}
