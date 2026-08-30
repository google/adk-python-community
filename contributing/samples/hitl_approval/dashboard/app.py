# Copyright 2026 Google LLC
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

"""Streamlit HITL Approval Dashboard.

Run:
    streamlit run contributing/samples/hitl_approval/dashboard/app.py
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"


def _resolve(request_id: str, decision: str, note: str):
    try:
        r = httpx.post(
            f"{API_BASE}/approvals/{request_id}/decide",
            json={
                "decision": decision,
                "reviewer_id": "dashboard_admin",
                "notes": note or None,
            },
            timeout=5,
        )
        r.raise_for_status()
        st.success(f"Request {request_id[:8]}… marked as {decision}.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to resolve: {e}")


st.set_page_config(page_title="ADK HITL Dashboard", page_icon="🔍", layout="wide")
st.title("ADK HITL Approval Dashboard")

# ── Sidebar filters ───────────────────────────────────────────────────────────

status_filter = st.sidebar.selectbox(
    "Filter by status", ["All", "pending", "approved", "rejected", "escalated"]
)

if st.sidebar.button("Refresh"):
    st.rerun()

# ── Fetch approvals ───────────────────────────────────────────────────────────

try:
    if status_filter == "pending":
        resp = httpx.get(f"{API_BASE}/approvals/pending", timeout=5)
    else:
        params = {}
        if status_filter != "All":
            params["decision"] = status_filter
        resp = httpx.get(f"{API_BASE}/approvals/audit", params=params, timeout=5)

    resp.raise_for_status()
    requests = resp.json()
except Exception as e:
    st.error(f"Could not connect to API: {e}")
    st.stop()

# ── Render approval cards ─────────────────────────────────────────────────────

if not requests:
    st.info("No approval requests found.")
else:
    for req in requests:
        status = req["status"]
        color = {
            "pending": "🟡",
            "approved": "🟢",
            "rejected": "🔴",
            "escalated": "🟠",
        }.get(status, "⚪")

        with st.expander(
            f"{color} [{status.upper()}] {req['tool_name']} — {req['agent_name']}  ({req['id'][:8]}…)"
        ):
            col1, col2 = st.columns(2)
            col1.markdown(
                f"**App:** `{req.get('app_name', 'N/A')}` | **User:** `{req.get('user_id', 'N/A')}`"
            )
            col1.markdown(f"**Agent:** `{req['agent_name']}`")
            col1.markdown(f"**Tool:** `{req['tool_name']}`")
            col1.markdown(f"**Session:** `{req['session_id']}`")
            col2.markdown(f"**Created:** {req['created_at']}")
            if req.get("decided_at"):
                col2.markdown(
                    f"**Resolved:** {req['decided_at']} by `{req.get('decided_by', 'unknown')}`"
                )

            st.markdown(f"**Message / Hint:**")
            st.info(req.get("message", "No message provided."))

            st.markdown("**Payload / Arguments:**")
            st.json(req.get("payload", {}))

            if req.get("decision_notes"):
                st.markdown(f"**Reviewer note:** {req['decision_notes']}")

            if status == "pending":
                note = st.text_input(
                    "Reviewer note (optional)", key=f"note_{req['id']}"
                )
                c1, c2, c3 = st.columns(3)

                if c1.button("Approve", key=f"approve_{req['id']}", type="primary"):
                    _resolve(req["id"], "approved", note)

                if c2.button("Reject", key=f"reject_{req['id']}"):
                    _resolve(req["id"], "rejected", note)

                if c3.button("Escalate", key=f"escalate_{req['id']}"):
                    _resolve(req["id"], "escalated", note)
