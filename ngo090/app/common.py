"""Pieces shared by every mode: loaders and the policy and cost settings."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The app lives in app/, the shared code in ../src: add the project root and this folder.
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

from src.artifacts import load_engine, load_run  # noqa: E402
from src.audit import AuditLog  # noqa: E402
from src.cases import CaseStore  # noqa: E402
from src.decision import PRESETS, CostConfig, Policy  # noqa: E402
from src.tools import Toolkit  # noqa: E402

MIN_EXPLAIN_RISK = 80.0     # the policy cannot flag anything below this risk (see the review slider)
CASES_DB = "FRAUD_CASES_DB"  # environment override, used by tests and for throwaway databases
AUDIT_DB = "FRAUD_AUDIT_DB"  # same, for the decision log


# --------------------------------------------------------------------------- #
# Cached loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the risk engine...")
def get_engine(tag: str):
    return load_engine(ROOT, tag)


@st.cache_data(show_spinner="Loading replay data...")
def get_run(tag: str):
    return load_run(ROOT, tag)


def cases_path() -> Path:
    """Where investigation cases live: data/cases.db unless the environment says otherwise."""
    override = os.environ.get(CASES_DB)
    return Path(override) if override else ROOT / "data" / "cases.db"


@st.cache_resource(show_spinner=False)
def get_case_store(path: str) -> CaseStore:
    """One store per database file, shared by every page and session."""
    return CaseStore(path)


def audit_path() -> Path:
    """Where the decision log lives: data/audit.db unless the environment says otherwise."""
    override = os.environ.get(AUDIT_DB)
    return Path(override) if override else ROOT / "data" / "audit.db"


@st.cache_resource(show_spinner=False)
def get_audit_log(path: str) -> AuditLog:
    """One audit log per database file: the live page writes, the audit page reads."""
    return AuditLog(path)


def audits() -> AuditLog:
    """The shared audit log at the default (or overridden) location."""
    return get_audit_log(str(audit_path()))


def toolkit(tag: str, policy: Policy | None = None, corpus=None) -> Toolkit:
    """The tool layer (src/tools.py), wired to what this app has already loaded.

    The pages call the same functions a script or a future agent calls; this
    only supplies the engine, the policy in the sidebar and the two stores at
    their configured paths.
    """
    return Toolkit(get_engine(tag), policy=policy, cases=get_case_store(str(cases_path())),
                   audit=get_audit_log(str(audit_path())), corpus=corpus, model_tag=tag)


# --------------------------------------------------------------------------- #
# Policy and cost settings (the same widgets in every mode)
# --------------------------------------------------------------------------- #
def popover():
    return st.popover if hasattr(st, "popover") else st.expander


def set_policy_widgets(block: float, capacity: int, review_min: int) -> None:
    st.session_state.block_risk = 100.0 if block >= 100 else float(block)
    st.session_state.capacity = int(capacity)
    st.session_state.review_min = int(review_min)


def apply_preset() -> None:
    preset = PRESETS.get(st.session_state.preset)
    if preset is not None:
        set_policy_widgets(preset.block_threshold, preset.daily_capacity, preset.review_min_score)


def mark_custom() -> None:
    st.session_state.preset = "Custom"              # a slider was moved by hand


def apply_recommendation() -> None:
    block, capacity = st.session_state.get("advisor_rec", (None, None))
    if block is not None:
        set_policy_widgets(block, capacity, st.session_state.get("review_min", 90))
        st.session_state.preset = "Custom"


def render_policy_popover() -> Policy:
    with popover()("Decision policy"):
        st.selectbox("Preset", [*PRESETS, "Custom"], key="preset", on_change=apply_preset,
                     help="Balanced is the default. Light touch raises fewer alerts, Strict raises more. "
                          "Moving a slider switches to Custom.")
        block_risk = st.slider("Auto-block above risk score", 90.0, 100.0, 99.5, 0.1, key="block_risk",
                               on_change=mark_custom,
                               help="Transactions at or above this risk are blocked automatically. 100 = never block.")
        capacity = st.slider("Daily review capacity", 0, 1000, 150, 10, key="capacity", on_change=mark_custom,
                             help="How many transactions the review team can check per simulated day.")
        review_min = st.slider("Only review above risk score", int(MIN_EXPLAIN_RISK), 99, 90, 1, key="review_min",
                               on_change=mark_custom, help="Reviewers skip transactions that look clearly normal.")
    return Policy(block_threshold=float("inf") if block_risk >= 100 else float(block_risk),
                  review_min_score=float(review_min), daily_capacity=int(capacity), capacity_mode="rolling")


def render_costs_popover() -> CostConfig:
    with popover()("Costs"):
        review_cost = st.slider("Cost of one review", 0, 20_000, 1_000, 100, key="review_cost")
        friction_cost = st.slider("Cost of blocking a legitimate customer", 0, 50_000, 5_000, 500, key="friction_cost")
        catch_rate = st.slider("Share of reviewed fraud analysts stop (%)", 0, 100, 100, 5, key="catch_rate")
    return CostConfig(review_cost=float(review_cost), friction_cost=float(friction_cost),
                      review_catch_rate=catch_rate / 100)