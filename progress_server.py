from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from threading import Lock
from datetime import datetime, timezone
import copy


# ============================================================
# QuantForge Progress Server
# ============================================================
#
# Purpose:
#   Receives progress updates from n8n and exposes the latest
#   valid state to the QuantForge frontend dashboard.
#
# Important:
#   Progress is MONOTONIC.
#
#   Once a research run reaches COMPLETED, later/stale progress
#   messages cannot move it backwards to CRITIC_RETRYING,
#   BACKTEST_STARTED, etc.
#
#   Similarly, an older experiment cannot overwrite a newer
#   experiment's state.
#
# ============================================================


app = FastAPI(
    title="QuantForge Progress Server",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IN-MEMORY STATE
# ============================================================

STATUS_STORE: Dict[str, Dict[str, Any]] = {}

STORE_LOCK = Lock()


# ============================================================
# STAGE ORDER
# ============================================================
#
# Higher number = further through the research process.
#
# Retry states intentionally share the Critic stage.
# This means:
#
#   CRITIC_STARTED
#       ↓
#   CRITIC_RETRYING
#       ↓
#   CRITIC_COMPLETE
#
# A retry cannot move a completed Critic stage backwards.
#
# ============================================================

STAGE_ORDER = {
    "RESEARCHER_STARTED": 1,
    "RESEARCHER_RETRYING": 1,
    "RESEARCHER_COMPLETE": 2,

    "BACKTEST_STARTED": 3,
    "BACKTEST_COMPLETE": 4,

    "HEAT_COMPLETE": 5,

    "CRITIC_STARTED": 6,
    "CRITIC_RETRYING": 6,
    "CRITIC_COMPLETE": 7,

    "VALIDATION_STARTED": 8,
    "VALIDATION_COMPLETE": 9,

    "REPORT_GENERATION": 10,

    "COMPLETED": 11,
}


# ============================================================
# TERMINAL STATES
# ============================================================

TERMINAL_STAGES = {
    "COMPLETED"
}


# ============================================================
# PYDANTIC MODEL
# ============================================================

class ProgressUpdate(BaseModel):

    research_run_id: str = Field(
        ...,
        min_length=1
    )

    status: str = "RUNNING"

    stage: str = ""

    message: str = ""

    experiment: Optional[int] = None

    max_experiments: Optional[int] = None

    strategy: Optional[str] = ""

    strategy_config: Dict[str, Any] = Field(
        default_factory=dict
    )

    metrics: Dict[str, Any] = Field(
        default_factory=dict
    )

    heat: Dict[str, Any] = Field(
        default_factory=dict
    )

    validation: Dict[str, Any] = Field(
        default_factory=dict
    )

    history: List[Any] = Field(
        default_factory=list
    )

    result: Any = None


# ============================================================
# HELPERS
# ============================================================

def normalize_stage(stage: Optional[str]) -> str:

    if not stage:
        return ""

    return str(stage).strip().upper()


def stage_rank(stage: Optional[str]) -> int:

    normalized = normalize_stage(stage)

    return STAGE_ORDER.get(
        normalized,
        0
    )


def safe_int(value: Any) -> Optional[int]:

    if value is None:
        return None

    try:
        return int(value)

    except (
        TypeError,
        ValueError
    ):
        return None


def make_state(
    update: ProgressUpdate
) -> Dict[str, Any]:

    state = update.model_dump()

    state["stage"] = normalize_stage(
        state.get("stage")
    )

    state["status"] = str(
        state.get("status") or "RUNNING"
    ).upper()

    state["updated_at"] = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    return state


# ============================================================
# PROGRESS DECISION
# ============================================================
#
# Determines whether a new progress event is allowed to
# replace the current state.
#
# ============================================================

def should_accept_update(
    current: Optional[Dict[str, Any]],
    incoming: Dict[str, Any]
):

    # --------------------------------------------------------
    # No existing state
    # --------------------------------------------------------

    if current is None:

        return True, "initial_state"


    current_stage = normalize_stage(
        current.get("stage")
    )

    incoming_stage = normalize_stage(
        incoming.get("stage")
    )


    # --------------------------------------------------------
    # 1. COMPLETED is terminal
    # --------------------------------------------------------
    #
    # Once completed, NOTHING can overwrite it.
    #
    # This directly fixes the problem we observed where:
    #
    # COMPLETED
    #     ↓
    # CRITIC_RETRYING
    #
    # caused the dashboard to go backwards.
    #
    # --------------------------------------------------------

    if current_stage in TERMINAL_STAGES:

        return (
            False,
            "current_state_is_terminal"
        )


    # --------------------------------------------------------
    # 2. Experiment comparison
    # --------------------------------------------------------

    current_experiment = safe_int(
        current.get("experiment")
    )

    incoming_experiment = safe_int(
        incoming.get("experiment")
    )


    # --------------------------------------------------------
    # Older experiment must never overwrite newer experiment
    # --------------------------------------------------------

    if (
        current_experiment is not None
        and incoming_experiment is not None
        and incoming_experiment < current_experiment
    ):

        return (
            False,
            "older_experiment"
        )


    # --------------------------------------------------------
    # Newer experiment is allowed to move forward even though
    # its stage rank may be numerically lower.
    #
    # Example:
    #
    # Experiment 1:
    #     CRITIC_COMPLETE
    #
    # Experiment 2:
    #     BACKTEST_STARTED
    #
    # This is VALID.
    # --------------------------------------------------------

    if (
        current_experiment is not None
        and incoming_experiment is not None
        and incoming_experiment > current_experiment
    ):

        return (
            True,
            "newer_experiment"
        )


    # --------------------------------------------------------
    # 3. Same experiment
    # --------------------------------------------------------

    if (
        current_experiment is not None
        and incoming_experiment is not None
        and incoming_experiment == current_experiment
    ):

        current_rank = stage_rank(
            current_stage
        )

        incoming_rank = stage_rank(
            incoming_stage
        )


        # ----------------------------------------------------
        # Ignore lower-stage updates
        # ----------------------------------------------------

        if incoming_rank < current_rank:

            return (
                False,
                "regressive_stage"
            )


        # ----------------------------------------------------
        # Same stage is allowed.
        #
        # Example:
        #
        # CRITIC_STARTED
        # CRITIC_RETRYING
        #
        # Both belong to the same stage.
        #
        # ----------------------------------------------------

        return (
            True,
            "same_or_forward_stage"
        )


    # --------------------------------------------------------
    # 4. If experiment numbers are unavailable
    # --------------------------------------------------------
    #
    # Fall back to stage ordering.
    #
    # --------------------------------------------------------

    current_rank = stage_rank(
        current_stage
    )

    incoming_rank = stage_rank(
        incoming_stage
    )


    if incoming_rank < current_rank:

        return (
            False,
            "regressive_stage_without_experiment"
        )


    return (
        True,
        "forward_stage_without_experiment"
    )


# ============================================================
# POST PROGRESS
# ============================================================

@app.post("/research-progress")
def update_progress(
    update: ProgressUpdate
):

    incoming = make_state(
        update
    )

    research_run_id = incoming[
        "research_run_id"
    ]


    with STORE_LOCK:

        current = STATUS_STORE.get(
            research_run_id
        )


        accepted, reason = should_accept_update(
            current,
            incoming
        )


        # ----------------------------------------------------
        # ACCEPT UPDATE
        # ----------------------------------------------------

        if accepted:

            STATUS_STORE[
                research_run_id
            ] = copy.deepcopy(
                incoming
            )

            stored = STATUS_STORE[
                research_run_id
            ]

            return {
                "ok": True,
                "accepted": True,
                "reason": reason,
                "research_run_id": research_run_id,
                "stage": stored.get("stage"),
                "status": stored.get("status"),
                "experiment": stored.get("experiment"),
                "max_experiments": stored.get(
                    "max_experiments"
                ),
                "last_update": stored.get(
                    "updated_at"
                ),
            }


        # ----------------------------------------------------
        # REJECT STALE UPDATE
        # ----------------------------------------------------
        #
        # Important:
        # We return HTTP 200.
        #
        # n8n therefore does NOT consider this a failed HTTP
        # request.
        #
        # The update is simply ignored because it is stale.
        #
        # ----------------------------------------------------

        return {
            "ok": True,
            "accepted": False,
            "reason": reason,
            "research_run_id": research_run_id,
            "current_stage": current.get(
                "stage"
            ) if current else None,
            "current_status": current.get(
                "status"
            ) if current else None,
            "current_experiment": current.get(
                "experiment"
            ) if current else None,
            "last_update": current.get(
                "updated_at"
            ) if current else None,
        }


# ============================================================
# GET CURRENT RESEARCH STATUS
# ============================================================

@app.get(
    "/research-status/{research_run_id}"
)
def get_research_status(
    research_run_id: str
):

    with STORE_LOCK:

        state = STATUS_STORE.get(
            research_run_id
        )


        if state is None:

            raise HTTPException(
                status_code=404,
                detail="Research run not found."
            )


        return copy.deepcopy(
            state
        )


# ============================================================
# DELETE A RESEARCH RUN
# ============================================================
#
# Useful for testing.
#
# Not required by the dashboard.
#
# ============================================================

@app.delete(
    "/research-status/{research_run_id}"
)
def delete_research_status(
    research_run_id: str
):

    with STORE_LOCK:

        if research_run_id not in STATUS_STORE:

            raise HTTPException(
                status_code=404,
                detail="Research run not found."
            )


        del STATUS_STORE[
            research_run_id
        ]


    return {
        "ok": True,
        "deleted": True,
        "research_run_id": research_run_id
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    with STORE_LOCK:

        active_runs = len(
            STATUS_STORE
        )


    return {
        "status": "QuantForge Progress Server is running",
        "version": "2.0.0",
        "active_runs": active_runs
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    with STORE_LOCK:

        active_runs = len(
            STATUS_STORE
        )


    return {
        "status": "healthy",
        "service": "QuantForge Progress Server",
        "active_runs": active_runs
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        reload=False
    )