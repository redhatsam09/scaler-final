from src.environment import EpisodeState, REASONING_MIN_CHARS


MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99


def _strict_task_score(value: float) -> float:
    if value <= MIN_TASK_SCORE:
        return MIN_TASK_SCORE
    if value >= MAX_TASK_SCORE:
        return MAX_TASK_SCORE
    return float(value)


def _action_count(episode_state: EpisodeState, action_name: str) -> int:
    return sum(1 for action in episode_state.actions_taken if action.get("action_type") == action_name)


def _has_action(episode_state: EpisodeState, action_name: str) -> bool:
    return _action_count(episode_state, action_name) > 0


def _loop_penalty(episode_state: EpisodeState) -> float:
    if len(episode_state.actions_taken) < 3:
        return 0.0
    penalties = 0.0
    for idx in range(2, len(episode_state.actions_taken)):
        triple = [
            episode_state.actions_taken[idx - 2].get("action_type"),
            episode_state.actions_taken[idx - 1].get("action_type"),
            episode_state.actions_taken[idx].get("action_type"),
        ]
        if len(set(triple)) == 1:
            penalties += 0.04
    return min(0.2, penalties)


def _excessive_deletion_penalty(episode_state: EpisodeState) -> float:
    original_len = len(episode_state.original_dataset)
    current_len = len(episode_state.dataset)
    if original_len <= 0:
        return 0.0
    deletion_ratio = max(0.0, (original_len - current_len) / original_len)
    if deletion_ratio <= 0.35:
        return 0.0
    return min(0.25, (deletion_ratio - 0.35) * 0.8)


def _kpi_component(episode_state: EpisodeState) -> float:
    if not episode_state.kpis:
        return 0.0
    quality = episode_state.kpis.get("quality_index", 0.0)
    compliance = episode_state.kpis.get("policy_compliance", 0.0)
    latency = episode_state.kpis.get("workflow_latency", 0.0)
    return min(1.0, max(0.0, quality * 0.45 + compliance * 0.35 + latency * 0.2))


def _actor_alignment_component(episode_state: EpisodeState) -> float:
    if not episode_state.kpis:
        return 0.0
    finance = episode_state.kpis.get("finance_cost_efficiency", 0.0)
    support = episode_state.kpis.get("support_sla_health", 0.0)
    sales = episode_state.kpis.get("sales_conversion_health", 0.0)
    return min(1.0, max(0.0, finance * 0.35 + support * 0.35 + sales * 0.3))


def _economic_penalty(episode_state: EpisodeState) -> float:
    overflow = max(0.0, episode_state.economic_cost_used - episode_state.economic_budget)
    if overflow <= 0:
        return 0.0
    return min(0.3, overflow / max(episode_state.economic_budget, 1.0))


def _stale_penalty(episode_state: EpisodeState) -> float:
    if episode_state.stale_penalty_active:
        return 0.08
    return 0.0


def _common_penalties(episode_state: EpisodeState) -> float:
    return _loop_penalty(episode_state) + _excessive_deletion_penalty(episode_state) + _economic_penalty(episode_state)


def _process_bonus(episode_state: EpisodeState) -> float:
    bonuses = episode_state.process_bonuses
    score = 0.0
    if bonuses.get("analyze_first"):
        score += 0.04
    if bonuses.get("post_drift_validate"):
        score += 0.06
    if bonuses.get("oversight_before_follow"):
        score += 0.06
    if bonuses.get("early_inspection"):
        score += 0.04
    return min(0.2, score)


def _reasoning_quality_penalty(episode_state: EpisodeState) -> float:
    penalty = 0.0
    for action in episode_state.actions_taken:
        reasoning = action.get("reasoning", "") or ""
        if isinstance(reasoning, str) and len(reasoning.strip()) < REASONING_MIN_CHARS:
            penalty += 0.01
    return min(0.1, penalty)


class MissingValuesGrader:
    @staticmethod
    def grade(episode_state: EpisodeState) -> float:
        dataset = episode_state.dataset
        original_dataset = episode_state.original_dataset
        original_missing = float(original_dataset.isnull().sum().sum())
        current_missing = float(dataset.isnull().sum().sum())

        if original_missing <= 0:
            base = 0.45
        else:
            base = max(0.0, (original_missing - current_missing) / original_missing)

        if _has_action(episode_state, "analyze"):
            base += 0.10
        if _has_action(episode_state, "impute"):
            base += 0.20
        if _has_action(episode_state, "validate"):
            base += 0.10
        if current_missing == 0:
            base += 0.15

        base += _kpi_component(episode_state) * 0.12
        base += _process_bonus(episode_state)
        base -= _common_penalties(episode_state)
        base -= _stale_penalty(episode_state)
        base -= _reasoning_quality_penalty(episode_state)
        return _strict_task_score(min(1.0, max(0.0, base)))


class DuplicateHandlingGrader:
    @staticmethod
    def grade(episode_state: EpisodeState) -> float:
        dataset = episode_state.dataset
        original_dataset = episode_state.original_dataset
        original_dups = float(original_dataset.duplicated(subset=None, keep=False).sum())
        current_dups = float(dataset.duplicated(subset=None, keep=False).sum())

        if original_dups <= 0:
            base = 0.4
        else:
            base = max(0.0, (original_dups - current_dups) / original_dups)

        if _has_action(episode_state, "deduplicate"):
            base += 0.2
        if _has_action(episode_state, "analyze"):
            base += 0.08
        if _has_action(episode_state, "validate"):
            base += 0.1
        if _has_action(episode_state, "report_findings"):
            base += 0.05
        if current_dups == 0:
            base += 0.15

        if not _has_action(episode_state, "analyze"):
            base -= 0.12
        if not _has_action(episode_state, "validate"):
            base -= 0.12

        base += _kpi_component(episode_state) * 0.1
        base += _process_bonus(episode_state)
        base -= _common_penalties(episode_state)
        base -= _stale_penalty(episode_state)
        base -= _reasoning_quality_penalty(episode_state)
        return _strict_task_score(min(1.0, max(0.0, base)))


class ComplexValidationGrader:
    @staticmethod
    def grade(episode_state: EpisodeState) -> float:
        dataset = episode_state.dataset
        original_dataset = episode_state.original_dataset

        original_missing = float(original_dataset.isnull().sum().sum())
        current_missing = float(dataset.isnull().sum().sum())
        missing_gain = (
            max(0.0, (original_missing - current_missing) / max(original_missing, 1.0)) if original_missing > 0 else 0.5
        )

        original_dups = float(original_dataset.duplicated(subset=None, keep=False).sum())
        current_dups = float(dataset.duplicated(subset=None, keep=False).sum())
        dup_gain = max(0.0, (original_dups - current_dups) / max(original_dups, 1.0)) if original_dups > 0 else 0.5

        action_diversity = len(set(action.get("action_type") for action in episode_state.actions_taken))
        diversity_score = min(1.0, action_diversity / 7.0)

        strategic_bonus = 0.0
        if _has_action(episode_state, "analyze"):
            strategic_bonus += 0.08
        if _has_action(episode_state, "validate"):
            strategic_bonus += 0.1
        if _has_action(episode_state, "report_findings"):
            strategic_bonus += 0.05
        if _has_action(episode_state, "reconcile_apps"):
            strategic_bonus += 0.12
        if episode_state.drift_active and _has_action(episode_state, "validate"):
            strategic_bonus += 0.08
        if episode_state.policy_version >= 3 and _has_action(episode_state, "oversight_review"):
            strategic_bonus += 0.06

        base = missing_gain * 0.25 + dup_gain * 0.2 + diversity_score * 0.2 + strategic_bonus
        base += _kpi_component(episode_state) * 0.12
        base += _actor_alignment_component(episode_state) * 0.08
        base += _process_bonus(episode_state)
        base -= _common_penalties(episode_state)
        base -= _stale_penalty(episode_state)
        base -= _reasoning_quality_penalty(episode_state)
        return _strict_task_score(min(1.0, max(0.0, base)))


class EnterpriseOrchestrationGrader:
    @staticmethod
    def grade(episode_state: EpisodeState) -> float:
        dataset = episode_state.dataset
        original_dataset = episode_state.original_dataset

        missing_before = float(original_dataset.isnull().sum().sum())
        missing_after = float(dataset.isnull().sum().sum())
        missing_gain = max(0.0, (missing_before - missing_after) / max(missing_before, 1.0))

        dup_before = float(original_dataset.duplicated(subset=None, keep=False).sum())
        dup_after = float(dataset.duplicated(subset=None, keep=False).sum())
        dup_gain = max(0.0, (dup_before - dup_after) / max(dup_before, 1.0))

        delegated = len(episode_state.delegated_work)
        resolved = sum(1 for status in episode_state.delegated_work.values() if status == "resolved")
        delegation_score = 0.0
        if delegated > 0:
            delegation_score = 0.35 + 0.65 * (resolved / delegated)

        drift_handled = 0.0
        if episode_state.drift_active and "compliance_tier" in dataset.columns:
            known = float((dataset["compliance_tier"] != "unknown").mean())
            drift_handled = known

        cross_app_alignment = 0.0
        if {"invoice_status", "ticket_status"}.issubset(dataset.columns):
            conflicts = ((dataset["invoice_status"] == "overdue") & (dataset["ticket_status"] == "resolved")).sum()
            cross_app_alignment = 1.0 - float(conflicts) / max(float(len(dataset)), 1.0)

        action_diversity = len(set(action.get("action_type") for action in episode_state.actions_taken))
        diversity = min(1.0, action_diversity / 8.0)

        oversight_component = 0.0
        if _has_action(episode_state, "oversight_review"):
            oversight_component += 0.08
        if episode_state.deception_detected:
            oversight_component += 0.12
        elif episode_state.deceptive_actor and not episode_state.deception_detected:
            oversight_component -= 0.08

        policy_adaptability = 0.0
        if episode_state.policy_version >= 2 and _has_action(episode_state, "validate"):
            policy_adaptability += 0.06
        if episode_state.policy_version >= 3 and _has_action(episode_state, "reconcile_apps"):
            policy_adaptability += 0.06

        base = (
            missing_gain * 0.16
            + dup_gain * 0.12
            + delegation_score * 0.16
            + drift_handled * 0.14
            + cross_app_alignment * 0.12
            + diversity * 0.07
            + _kpi_component(episode_state) * 0.12
            + _actor_alignment_component(episode_state) * 0.08
            + oversight_component
            + policy_adaptability
        )
        if _has_action(episode_state, "report_findings"):
            base += 0.04
        if _has_action(episode_state, "inspect_actor"):
            base += 0.05
        if len(episode_state.inspected_actors) >= 2:
            base += 0.04
        contested = sum(1 for v in episode_state.delegated_work.values() if v == "contested")
        resolved_contested = sum(1 for v in episode_state.delegated_work.values() if v == "resolved")
        if contested > 0 and resolved_contested > 0:
            base += 0.06  # Handled pushback gracefully
        base += _process_bonus(episode_state)
        base -= _common_penalties(episode_state)
        base -= _stale_penalty(episode_state)
        base -= _reasoning_quality_penalty(episode_state)
        return _strict_task_score(min(1.0, max(0.0, base)))

