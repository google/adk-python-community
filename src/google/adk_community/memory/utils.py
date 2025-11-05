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


def extract_text_from_event(event) -> str:
  """Extracts text content from an event's content parts.

  Filters out thought parts and only extracts actual text content.
  This ensures metadata like thoughtSignature is not stored in memories.

  Args:
      event: The event to extract text from.

  Returns:
      Combined text from all text parts (excluding thoughts), or empty string if none found.
  """
  if not event.content or not event.content.parts:
    return ''

  # Filter out thought parts and only extract text
  # This prevents metadata like thoughtSignature from being stored
  text_parts = [
      part.text 
      for part in event.content.parts 
      if part.text and not part.thought
  ]
  return ' '.join(text_parts)

