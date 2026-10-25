"""Admin sign-in for the dashboard. One shared AdminAuth (PBKDF2 hash, lockout after 3 failures) guards everything that shows or
downloads the ledger / blockchain / calibration records, and the 'deal with a quarantined sensor' dialog.

The secret comes from the IOTGUARD_ADMIN_SECRET environment variable; without it the demo secret below is used (fine for a
demo on your own laptop; never reuse a real password here)."""
from __future__ import annotations

import os
import time

import streamlit as st

from iotguard.security import AdminAuth

DEFAULT_SECRET = "demo-admin-secret"
SESSION_MINUTES = 30


def server_secret() -> str:
    return os.environ.get("IOTGUARD_ADMIN_SECRET") or DEFAULT_SECRET


@st.cache_resource(show_spinner=False)
def auth_service() -> AdminAuth:
    """One per server: the lockout counter is shared by every browser tab."""
    return AdminAuth(server_secret(), max_failures=3, lockout_s=120.0)


def using_default_secret() -> bool:
    return server_secret() == DEFAULT_SECRET


def is_admin() -> bool:
    ok = st.session_state.get("_admin_until", 0.0) > time.time()
    if not ok:
        st.session_state.pop("_admin_secret", None)
    return ok


def secret() -> str | None:
    """The secret typed at sign-in (kept in this browser session only, never written anywhere); the engine re-checks it per action."""
    return st.session_state.get("_admin_secret") if is_admin() else None


def sign_out() -> None:
    st.session_state.pop("_admin_until", None)
    st.session_state.pop("_admin_secret", None)


def try_sign_in(typed: str) -> tuple[bool, str]:
    a = auth_service()
    if a.locked:
        return False, "Too many wrong attempts. Locked for 2 minutes."
    if a.check(typed, None):
        st.session_state["_admin_until"] = time.time() + SESSION_MINUTES * 60
        st.session_state["_admin_secret"] = typed
        return True, ""
    return False, "Wrong admin secret." + (" Locked for 2 minutes." if a.locked else "")


def login_form(key: str = "admin_login", hint: bool = True, in_dialog: bool = False) -> bool:
    """Draws the sign-in form; returns True once signed in (and reruns)."""
    if is_admin():
        return True
    with st.form(key, clear_on_submit=True):
        typed = st.text_input("Admin secret", type="password", key=f"{key}_pw")
        go = st.form_submit_button("Sign in", type="primary")
    if hint and using_default_secret():
        st.caption("Demo build: the admin secret is `demo-admin-secret`.")
    if go:
        ok, msg = try_sign_in(typed)
        if ok:
            st.rerun(scope="fragment" if in_dialog else "app")
        st.error(msg)
    return False


def require_admin(what: str) -> None:
    """Call at the top of anything restricted: shows a lock + sign-in and stops the page for non-admins."""
    if is_admin():
        return
    st.markdown(f"#### 🔒 {what}")
    st.info("Only the admin can see this. The simulator and the live sensor values are open to everyone.")
    login_form(f"gate_{abs(hash(what)) % 10_000}")
    st.stop()


def sidebar_status() -> None:
    if is_admin():
        st.sidebar.success("🔓 Admin signed in")
        if st.sidebar.button("Sign out", key="sb_signout"):
            sign_out()
            st.rerun()
    else:
        st.sidebar.caption("🔒 Not signed in as admin (ledger, blocks and downloads are locked).")
