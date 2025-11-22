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

"""Content moderator bot example using DjangoInstructionProvider.

This example demonstrates how to use Django templating for dynamic agent
instructions that leverage Django's familiar syntax and built-in filters.

The agent reviews user-generated content and provides moderation decisions,
showcasing Django template features like filters and for...empty tags.

Usage:
    adk run contributing/samples/templating/django_content_moderator
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk_community.templating import DjangoInstructionProvider


def populate_moderation_queue(callback_context: CallbackContext):
  """Populate the session state with content moderation queue.

  In a real application, this would fetch content from a moderation system.
  For this demo, we use sample data.
  """
  callback_context.state['platform_name'] = 'SocialHub'
  callback_context.state['moderator_name'] = 'Alex Chen'

  callback_context.state['pending_content'] = [
      {
          'id': 'POST-001',
          'type': 'Comment',
          'author': 'user123',
          'text': 'Great product! Highly recommended.',
          'reports': 0,
          'flagged_words': [],
      },
      {
          'id': 'POST-002',
          'type': 'Review',
          'author': 'angry_customer',
          'text': 'This is terrible service! Very disappointed.',
          'reports': 3,
          'flagged_words': ['terrible'],
      },
      {
          'id': 'POST-003',
          'type': 'Forum Post',
          'author': 'spammer99',
          'text': 'Click here for free stuff!!!',
          'reports': 5,
          'flagged_words': ['click here', 'free'],
      },
  ]

  callback_context.state['moderation_stats'] = {
      'total_reviewed_today': 47,
      'approved': 32,
      'rejected': 10,
      'pending': 5,
  }


# Define the Django template for instructions
# Django uses {{ }} for variables, {% %} for tags, and | for filters
CONTENT_MODERATOR_TEMPLATE = """
You are a Content Moderation Agent for {{ platform_name }}.
Moderator: {{ moderator_name }}

MODERATION QUEUE
================
{% if pending_content %}
{{ pending_content|length }} item{{ pending_content|length|pluralize }} pending review:

{% for item in pending_content %}
[{{ item.id }}] {{ item.type }} by @{{ item.author }}
Content: "{{ item.text|truncatewords:15 }}"
Reports: {{ item.reports }}
{% if item.flagged_words %}
⚠️  Flagged Keywords: {{ item.flagged_words|join:", " }}
{% endif %}
{% if item.reports > 2 %}
🔴 HIGH PRIORITY - Multiple user reports!
{% elif item.reports > 0 %}
⚠️  MEDIUM PRIORITY - User reported
{% else %}
✅ LOW PRIORITY - No reports
{% endif %}
---
{% endfor %}
{% else %}
✅ Queue is clear! No content pending moderation.
{% endif %}

TODAY'S MODERATION STATS
========================
Total Reviewed: {{ moderation_stats.total_reviewed_today }}
Approved: {{ moderation_stats.approved }} ({{ moderation_stats.approved|add:0|floatformat:0 }})
Rejected: {{ moderation_stats.rejected }}
Still Pending: {{ moderation_stats.pending }}

INSTRUCTIONS
============
Review each piece of content and determine if it should be:
1. **APPROVED** - Complies with community guidelines
2. **REJECTED** - Violates guidelines (spam, harassment, etc.)
3. **FLAGGED FOR HUMAN REVIEW** - Edge case requiring manual review

Guidelines:
- Multiple reports or flagged keywords warrant careful review
- Consider context, not just keywords
- Err on the side of caution for borderline cases
- Be fair and consistent
- Provide clear reasoning for decisions

{% if pending_content %}
Prioritize items with the most reports first.
{% endif %}

Be objective, thorough, and protect the community while respecting free expression.
""".strip()

# Create the provider with the template
instruction_provider = DjangoInstructionProvider(CONTENT_MODERATOR_TEMPLATE)

# Create the agent
root_agent = Agent(
    name='content_moderator_agent',
    model='gemini-2.0-flash',
    instruction=instruction_provider,
    before_agent_callback=[populate_moderation_queue],
)
