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

"""Example of using Redis for Session Service with FastAPI."""

import os
import uvicorn

from google.adk.cli.fast_api import get_fast_api_app

AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "redis_service_agent")
SESSION_SERVICE_URI = os.environ.get("REDIS_URI", "redis://127.0.0.1:6379/0")
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
SERVE_WEB_INTERFACE = True
ARTIFACT_SERVICE_URI = ""


def main():
    """Main function to run the FastAPI application."""
    app = get_fast_api_app(
        agents_dir=AGENTS_DIR,
        session_service_uri=SESSION_SERVICE_URI,
        session_db_kwargs=dict(expire=60 * 60 * 1),  # 1 hour of Time To Live
        artifact_service_uri=ARTIFACT_SERVICE_URI,
        allow_origins=ALLOWED_ORIGINS,
        web=SERVE_WEB_INTERFACE,
    )

    @app.get("/health")
    async def get_health():
        return {"status": "ok"}

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


if __name__ == "__main__":
    main()
