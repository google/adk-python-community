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

"""Templating engine integrations for ADK instruction providers.

This package provides InstructionProvider implementations for popular
Python templating engines, allowing developers to use their preferred
templating syntax for dynamic agent instructions.

Available Providers:
    - Jinja2InstructionProvider: Jinja2 templating (most popular)
    - MakoInstructionProvider: Mako templating (fast, Python-centric)
    - MustacheInstructionProvider: Mustache/Chevron (logic-less)
    - DjangoInstructionProvider: Django templates

Installation:
    pip install google-adk-community[templating]

Example:
    ```python
    from google.adk.agents import Agent
    from google.adk_community.templating import Jinja2InstructionProvider

    provider = Jinja2InstructionProvider('''
        You are a {{ role }} assistant.
        {% if user %}
        Current user: {{ user.name }}
        {% endif %}
    ''')

    agent = Agent(
        name="my_agent",
        model="gemini-2.0-flash",
        instruction=provider
    )
    ```
"""

from .base import BaseTemplateProvider
from .django_provider import DjangoInstructionProvider
from .jinja2_provider import Jinja2InstructionProvider
from .mako_provider import MakoInstructionProvider
from .mustache_provider import MustacheInstructionProvider

__all__ = [
    'BaseTemplateProvider',
    'Jinja2InstructionProvider',
    'MakoInstructionProvider',
    'MustacheInstructionProvider',
    'DjangoInstructionProvider',
]
