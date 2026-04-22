from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Observation(BaseModel):
    dataset_shape: tuple[int, int] = Field(description="Current dataset shape (rows, columns)")
    column_names: List[str] = Field(description="List of column names in dataset")
    data_types: Dict[str, str] = Field(description="Data type for each column")
    missing_values: Dict[str, int] = Field(description="Count of missing values per column")
    current_state: str = Field(description="Human-readable description of current data state")
    task_id: str = Field(description="Current task being evaluated")
    step_count: int = Field(description="Current step number in episode")
    episode_progress: str = Field(description="Summary of progress made so far")
    drift_notice: Optional[str] = Field(
        default=None,
        description="Schema/policy drift notice when environment contracts change mid-episode",
    )
    actor_messages: List[str] = Field(
        default_factory=list,
        description="Recent messages from simulated actors in the environment",
    )
    actor_objectives: Dict[str, str] = Field(
        default_factory=dict,
        description="Current actor incentives/objectives in the simulated enterprise",
    )
    actor_conflicts: Dict[str, str] = Field(
        default_factory=dict,
        description="Explicit incentive conflicts that create multi-agent negotiation pressure",
    )
    kpi_snapshot: Dict[str, float] = Field(
        default_factory=dict,
        description="Current high-level KPI metrics visible to the agent",
    )
    policy_version: int = Field(
        default=1,
        description="Current policy/T&C version enforced by the environment",
    )
    difficulty: str = Field(
        default="medium",
        description="Curriculum difficulty level for the episode: easy, medium, or hard",
    )
    economic_status: Dict[str, float] = Field(
        default_factory=dict,
        description="Economic trade-off context such as cost used, budget, and cost efficiency",
    )


class Action(BaseModel):
    action_type: str = Field(
        description=(
            "Type of action: analyze, impute, deduplicate, validate, report_findings, "
            "delegate, resolve_alert, reconcile_apps, or oversight_review"
        )
    )
    target_columns: List[str] = Field(description="Columns to apply action on")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the action"
    )
    reasoning: str = Field(description="Explanation of why this action was chosen")


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0, description="Reward value between 0.0 and 1.0")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward components"
    )
    message: str = Field(description="Explanation of reward calculation")
