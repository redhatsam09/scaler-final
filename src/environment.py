import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models import Action, Observation, Reward


DEFAULT_ENV_SEED = 42


@dataclass
class EpisodeState:
    dataset: pd.DataFrame
    original_dataset: pd.DataFrame
    task_id: str
    dataset_name: str
    seed: int
    step_count: int = 0
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)
    drift_active: bool = False
    drift_notice: Optional[str] = None
    actor_inbox: List[str] = field(default_factory=list)
    delegated_work: Dict[str, str] = field(default_factory=dict)
    kpis: Dict[str, float] = field(default_factory=dict)


class DataCleaningEnv:
    TASK_TEMPLATE_MAP = {
        "task_missing_values": "crm_contacts",
        "task_duplicate_handling": "billing_invoices",
        "task_complex_validation": "support_tickets",
        "task_enterprise_orchestration": "enterprise_orchestration",
    }

    VALID_ACTIONS = {
        "analyze",
        "impute",
        "deduplicate",
        "validate",
        "report_findings",
        "delegate",
        "resolve_alert",
        "reconcile_apps",
    }

    def __init__(self, seed: int = DEFAULT_ENV_SEED):
        self.current_episode: Optional[EpisodeState] = None
        self.max_steps = 60
        self.default_seed = int(seed)
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._rng = random.Random(self.seed)
        self.dataset_templates = self._create_dataset_templates()

    def _create_dataset_templates(self) -> Dict[str, pd.DataFrame]:
        templates: Dict[str, pd.DataFrame] = {}
        rng = random.Random(self.seed)

        crm_data = pd.DataFrame(
            {
                "contact_id": list(range(1, 121)),
                "account_id": [f"ACC{1000 + i}" for i in range(120)],
                "name": [f"Contact_{i}" if rng.random() > 0.13 else None for i in range(120)],
                "email": [f"contact{i}@example.com" if rng.random() > 0.17 else None for i in range(120)],
                "phone": [f"+1-202-555-{1000 + i}" if rng.random() > 0.19 else None for i in range(120)],
                "lead_source": [rng.choice(["website", "partner", "event", "outbound", None]) for _ in range(120)],
                "country": [rng.choice(["US", "UK", "CA", "DE", None]) for _ in range(120)],
            }
        )
        crm_data = pd.concat([crm_data, crm_data.iloc[:12]], ignore_index=True)
        templates["crm_contacts"] = crm_data.reset_index(drop=True)

        billing_data = pd.DataFrame(
            {
                "invoice_id": [f"INV{5000 + i}" for i in range(160)],
                "account_id": [f"ACC{1000 + (i % 90)}" for i in range(160)],
                "amount": [round(rng.uniform(100, 15000), 2) if rng.random() > 0.08 else None for _ in range(160)],
                "currency": [rng.choice(["USD", "EUR", "GBP", None]) for _ in range(160)],
                "status": [rng.choice(["paid", "pending", "overdue", None]) for _ in range(160)],
                "due_date": pd.date_range("2024-01-01", periods=160, freq="D").astype(str),
                "paid_date": [
                    str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i + 5)) if rng.random() > 0.3 else None
                    for i in range(160)
                ],
            }
        )
        billing_data = pd.concat([billing_data, billing_data.iloc[:18]], ignore_index=True)
        templates["billing_invoices"] = billing_data.reset_index(drop=True)

        support_data = pd.DataFrame(
            {
                "ticket_id": [f"TKT{9000 + i}" for i in range(180)],
                "account_id": [f"ACC{1000 + (i % 95)}" for i in range(180)],
                "priority": [rng.choice(["low", "medium", "high", "critical", None]) for _ in range(180)],
                "status": [rng.choice(["new", "in_progress", "blocked", "resolved", None]) for _ in range(180)],
                "opened_at": pd.date_range("2024-03-01", periods=180, freq="8h").astype(str),
                "resolved_at": [
                    str(pd.Timestamp("2024-03-01") + pd.Timedelta(hours=8 * (i + 2))) if rng.random() > 0.4 else None
                    for i in range(180)
                ],
                "csat_score": [round(rng.uniform(1.0, 5.0), 2) if rng.random() > 0.25 else None for _ in range(180)],
                "agent": [rng.choice(["alice", "bob", "carol", "dave", None]) for _ in range(180)],
            }
        )
        support_data = pd.concat([support_data, support_data.iloc[:15]], ignore_index=True)
        templates["support_tickets"] = support_data.reset_index(drop=True)

        enterprise = pd.DataFrame(
            {
                "workflow_id": [f"WF-{10000 + i}" for i in range(240)],
                "account_id": [f"ACC{1000 + (i % 95)}" for i in range(240)],
                "crm_email": [f"user{i}@example.com" if rng.random() > 0.22 else None for i in range(240)],
                "crm_owner": [rng.choice(["alice", "bob", "carol", "dave", None]) for _ in range(240)],
                "invoice_id": [f"INV{5000 + (i % 170)}" for i in range(240)],
                "invoice_amount": [round(rng.uniform(100, 20000), 2) if rng.random() > 0.1 else None for _ in range(240)],
                "invoice_status": [rng.choice(["paid", "pending", "overdue", None]) for _ in range(240)],
                "ticket_id": [f"TKT{9000 + (i % 190)}" for i in range(240)],
                "ticket_priority": [rng.choice(["low", "medium", "high", "critical", None]) for _ in range(240)],
                "ticket_status": [rng.choice(["new", "in_progress", "blocked", "resolved", None]) for _ in range(240)],
                "sla_hours": [rng.choice([4, 8, 24, 48, None]) for _ in range(240)],
                "region": [rng.choice(["US", "EU", "APAC", None]) for _ in range(240)],
            }
        )
        enterprise = pd.concat([enterprise, enterprise.iloc[:30]], ignore_index=True)
        templates["enterprise_orchestration"] = enterprise.reset_index(drop=True)
        return templates

    def _set_seed(self, seed: int) -> None:
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._rng = random.Random(self.seed)
        self.dataset_templates = self._create_dataset_templates()

    def _select_template_name(self, task_id: str) -> str:
        mapped = self.TASK_TEMPLATE_MAP.get(task_id)
        if mapped and mapped in self.dataset_templates:
            return mapped
        return self._rng.choice(list(self.dataset_templates.keys()))

    def _initial_kpis(self, dataset: pd.DataFrame) -> Dict[str, float]:
        rows = float(len(dataset))
        missing_total = float(dataset.isnull().sum().sum())
        duplicates_total = float(dataset.duplicated(subset=None, keep=False).sum())
        denominator = max(rows * max(float(len(dataset.columns)), 1.0), 1.0)
        quality_index = max(0.0, 1.0 - (missing_total / denominator) - (duplicates_total / max(rows, 1.0)) * 0.1)
        return {
            "quality_index": round(min(1.0, quality_index), 6),
            "backlog_pressure": round(min(1.0, missing_total / max(rows * 0.4, 1.0)), 6),
            "policy_compliance": 0.5,
            "workflow_latency": 1.0,
        }

    def reset(self, task_id: str = "task_missing_values", seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._set_seed(seed)

        template_name = self._select_template_name(task_id)
        selected_template = self.dataset_templates[template_name]
        dataset = selected_template.copy(deep=True)
        episode = EpisodeState(
            dataset=dataset,
            original_dataset=dataset.copy(deep=True),
            task_id=task_id,
            dataset_name=template_name,
            seed=self.seed,
            step_count=0,
        )
        episode.kpis = self._initial_kpis(dataset)
        if task_id == "task_enterprise_orchestration":
            episode.actor_inbox = [
                "Sales Ops: lead ownership conflicts in EU accounts.",
                "Finance Bot: duplicate invoice IDs detected in overdue queue.",
                "Support Lead: critical tickets missing SLA assignments.",
            ]
        self.current_episode = episode
        return self._get_observation()

    def _schema_drift_trigger(self) -> None:
        episode = self.current_episode
        if not episode:
            return
        if episode.task_id != "task_enterprise_orchestration":
            return
        if episode.drift_active:
            return
        if episode.step_count < 4:
            return

        dataset = episode.dataset
        if "compliance_tier" not in dataset.columns:
            dataset["compliance_tier"] = dataset["region"].map(
                {"EU": "strict", "US": "standard", "APAC": "expanded"}
            ).fillna("unknown")
        dataset["invoice_status"] = dataset["invoice_status"].replace({"pending": "awaiting_payment"})
        episode.drift_active = True
        episode.drift_notice = (
            "Schema drift: added compliance_tier and replaced invoice_status 'pending' with "
            "'awaiting_payment'. Validation policy now requires compliance_tier checks."
        )
        episode.actor_inbox.append("Compliance Officer: all EU workflows now require strict tier confirmation.")

    def _update_kpis(self) -> None:
        episode = self.current_episode
        if not episode:
            return
        dataset = episode.dataset
        rows = float(len(dataset))
        missing_total = float(dataset.isnull().sum().sum())
        duplicates_total = float(dataset.duplicated(subset=None, keep=False).sum())
        denominator = max(rows * max(float(len(dataset.columns)), 1.0), 1.0)
        quality_index = max(0.0, 1.0 - (missing_total / denominator) - (duplicates_total / max(rows, 1.0)) * 0.12)

        latency = max(0.0, 1.0 - min(episode.step_count / max(self.max_steps, 1), 1.0) * 0.7)
        compliance = 0.5
        if episode.drift_active and "compliance_tier" in dataset.columns:
            compliance = 0.4 + float((dataset["compliance_tier"] != "unknown").mean()) * 0.6
        elif episode.task_id != "task_enterprise_orchestration":
            compliance = 0.75

        delegated_resolved = sum(1 for status in episode.delegated_work.values() if status == "resolved")
        backlog_pressure = max(0.0, 1.0 - delegated_resolved * 0.08)

        episode.kpis = {
            "quality_index": round(min(1.0, quality_index), 6),
            "backlog_pressure": round(min(1.0, max(0.0, backlog_pressure)), 6),
            "policy_compliance": round(min(1.0, max(0.0, compliance)), 6),
            "workflow_latency": round(min(1.0, max(0.0, latency)), 6),
        }

    def _describe_state(self) -> str:
        if not self.current_episode:
            return "No active episode"
        episode = self.current_episode
        rows, cols = episode.dataset.shape
        missing_count = int(episode.dataset.isnull().sum().sum())
        dup_count = int(episode.dataset.duplicated(subset=None, keep=False).sum())
        kpi_summary = ", ".join(f"{k}={v:.3f}" for k, v in episode.kpis.items())
        return (
            f"Dataset {episode.dataset_name} ({rows} rows, {cols} cols): {missing_count} missing values, "
            f"{dup_count} potential duplicates, KPIs[{kpi_summary}]"
        )

    def _get_progress_summary(self) -> str:
        if not self.current_episode:
            return "No progress"
        episode = self.current_episode
        if not episode.actions_taken:
            return "No actions taken yet"
        recent = ", ".join(a.get("action_type", "?") for a in episode.actions_taken[-4:])
        return f"Completed {len(episode.actions_taken)} action(s): {recent}"

    def _get_observation(self) -> Observation:
        if not self.current_episode:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        episode = self.current_episode
        missing_values = episode.dataset.isnull().sum().to_dict()
        return Observation(
            dataset_shape=tuple(episode.dataset.shape),
            column_names=list(episode.dataset.columns),
            data_types={col: str(dtype) for col, dtype in episode.dataset.dtypes.items()},
            missing_values=missing_values,
            current_state=self._describe_state(),
            task_id=episode.task_id,
            step_count=episode.step_count,
            episode_progress=self._get_progress_summary(),
            drift_notice=episode.drift_notice,
            actor_messages=episode.actor_inbox[-3:],
            kpi_snapshot=episode.kpis,
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if not self.current_episode:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        episode = self.current_episode
        episode.step_count += 1
        episode.actions_taken.append(
            {
                "action_type": action.action_type,
                "target_columns": action.target_columns,
                "parameters": action.parameters,
                "step": episode.step_count,
            }
        )
        self._schema_drift_trigger()
        reward, info = self._process_action(action)
        self._update_kpis()

        done = episode.step_count >= self.max_steps
        if action.action_type == "report_findings" and episode.step_count >= 6:
            done = True
        observation = self._get_observation()
        info["kpi_snapshot"] = dict(episode.kpis)
        info["drift_active"] = episode.drift_active
        return observation, reward, done, info

    def _invalid_action_penalty(self, action_type: str) -> float:
        if action_type in self.VALID_ACTIONS:
            return 0.0
        return 0.25

    def _repeat_action_penalty(self) -> float:
        episode = self.current_episode
        if not episode or len(episode.actions_taken) < 3:
            return 0.0
        last_three = [a["action_type"] for a in episode.actions_taken[-3:]]
        if len(set(last_three)) == 1:
            return 0.2
        return 0.0

    def _process_action(self, action: Action) -> Tuple[Reward, Dict[str, Any]]:
        if not self.current_episode:
            raise RuntimeError("No active episode")
        components: Dict[str, float] = {}
        messages: List[str] = []
        episode = self.current_episode
        dataset_before = episode.dataset.copy(deep=False)
        missing_before = float(dataset_before.isnull().sum().sum())
        dup_before = float(dataset_before.duplicated(subset=None, keep=False).sum())

        if action.action_type == "analyze":
            components["analysis"] = self._perform_analysis(action.target_columns)
            messages.append(f"Analyzed {len(action.target_columns)} columns")
        elif action.action_type == "impute":
            components["imputation"] = self._perform_imputation(action.target_columns, action.parameters)
            messages.append("Imputation applied")
        elif action.action_type == "deduplicate":
            components["deduplication"] = self._perform_deduplication(action.parameters)
            messages.append("Deduplication executed")
        elif action.action_type == "validate":
            components["validation"] = self._perform_validation(action.target_columns, action.parameters)
            messages.append("Validation performed")
        elif action.action_type == "report_findings":
            components["reporting"] = self._generate_report(action.parameters)
            messages.append("Report generated")
        elif action.action_type == "delegate":
            components["delegation"] = self._perform_delegation(action.parameters)
            messages.append("Delegated work to actor")
        elif action.action_type == "resolve_alert":
            components["alert_resolution"] = self._perform_alert_resolution(action.parameters)
            messages.append("Resolved actor escalation")
        elif action.action_type == "reconcile_apps":
            components["reconciliation"] = self._perform_reconciliation(action.parameters)
            messages.append("Reconciled records across app surfaces")
        else:
            components["invalid_action"] = 0.0
            messages.append(f"Unknown action type: {action.action_type}")

        missing_after = float(episode.dataset.isnull().sum().sum())
        dup_after = float(episode.dataset.duplicated(subset=None, keep=False).sum())
        delta_missing = max(0.0, missing_before - missing_after)
        delta_dups = max(0.0, dup_before - dup_after)
        shaping = min(1.0, (delta_missing / max(missing_before, 1.0)) * 0.6 + (delta_dups / max(dup_before, 1.0)) * 0.4)
        components["progress_signal"] = shaping

        invalid_penalty = self._invalid_action_penalty(action.action_type)
        repeat_penalty = self._repeat_action_penalty()
        if invalid_penalty > 0:
            components["invalid_penalty"] = -invalid_penalty
            messages.append("Penalty: invalid action")
        if repeat_penalty > 0:
            components["loop_penalty"] = -repeat_penalty
            messages.append("Penalty: repeated same action type")

        raw_total = sum(components.values()) / max(len(components), 1)
        total_reward = min(1.0, max(0.0, raw_total))
        info = {
            "action_type": action.action_type,
            "reasoning": action.reasoning,
            "components": components,
            "messages": messages,
        }
        return Reward(value=total_reward, components=components, message="; ".join(messages)), info

    def _perform_analysis(self, columns: List[str]) -> float:
        if not self.current_episode:
            return 0.0
        dataset = self.current_episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        if not valid_cols:
            return 0.0
        reward = 0.0
        for col in valid_cols:
            missing_pct = float(dataset[col].isnull().sum()) / max(float(len(dataset)), 1.0)
            if missing_pct > 0:
                reward += 0.25
            if dataset[col].dtype == "object":
                reward += 0.1
            if dataset[col].duplicated().sum() > 0:
                reward += 0.1
        return min(1.0, reward / len(valid_cols))

    def _perform_imputation(self, columns: List[str], params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        dataset = self.current_episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        if not valid_cols:
            return 0.0
        reward = 0.0
        method = params.get("method", "mean")
        for col in valid_cols:
            before_missing = int(dataset[col].isnull().sum())
            if before_missing == 0:
                reward += 0.03
                continue
            if method in {"mean", "median"} and pd.api.types.is_numeric_dtype(dataset[col]):
                value = dataset[col].mean() if method == "mean" else dataset[col].median()
                if pd.notna(value):
                    dataset[col] = dataset[col].fillna(value)
            elif method in {"forward_fill", "ffill"}:
                dataset[col] = dataset[col].ffill().bfill()
            elif method == "mode":
                mode = dataset[col].mode(dropna=True)
                if not mode.empty:
                    dataset[col] = dataset[col].fillna(mode.iloc[0])
            after_missing = int(dataset[col].isnull().sum())
            improvement = max(0.0, float(before_missing - after_missing) / max(before_missing, 1))
            reward += 0.1 + improvement * 0.5
        return min(1.0, reward / len(valid_cols))

    def _perform_deduplication(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        dataset = self.current_episode.dataset
        dup_before = int(dataset.duplicated(subset=None, keep=False).sum())
        if dup_before == 0:
            return 0.2
        subset = params.get("subset")
        keep = params.get("keep", "first")
        # Duplicate-handling task is intentionally strict to prevent reward gaming via broad row drops.
        if (
            self.current_episode.task_id == "task_duplicate_handling"
            and (not isinstance(subset, list) or "invoice_id" not in subset)
        ):
            return 0.05

        if subset and isinstance(subset, list) and all(c in dataset.columns for c in subset):
            dataset.drop_duplicates(subset=subset, keep=keep, inplace=True)
        else:
            dataset.drop_duplicates(keep=keep, inplace=True)
        dup_after = int(dataset.duplicated(subset=None, keep=False).sum())
        if dup_before <= 0:
            return 0.0
        return min(1.0, max(0.0, 0.2 + (dup_before - dup_after) / dup_before * 0.8))

    def _perform_validation(self, columns: List[str], params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        dataset = self.current_episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        if not valid_cols:
            return 0.0
        reward = 0.0
        for col in valid_cols:
            validation_type = params.get(f"{col}_type", "exists")
            if validation_type == "exists":
                reward += 0.5 if dataset[col].isnull().sum() == 0 else 0.15
            elif validation_type == "numeric":
                reward += 0.5 if pd.api.types.is_numeric_dtype(dataset[col]) else 0.05
            elif validation_type == "range":
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    min_val = params.get(f"{col}_min", dataset[col].min())
                    max_val = params.get(f"{col}_max", dataset[col].max())
                    in_range = ((dataset[col] >= min_val) & (dataset[col] <= max_val)).sum()
                    reward += (in_range / max(len(dataset), 1)) * 0.5
            elif validation_type == "categorical_nonempty":
                non_empty = dataset[col].notna().sum()
                reward += (non_empty / max(len(dataset), 1)) * 0.5

        if self.current_episode.drift_active and "compliance_tier" in dataset.columns:
            compliance_ok = float((dataset["compliance_tier"] != "unknown").mean())
            reward += 0.2 * compliance_ok

        return min(1.0, reward / len(valid_cols))

    def _generate_report(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        episode = self.current_episode
        dataset = episode.dataset
        original = episode.original_dataset
        reward = 0.0
        if params.get("include_summary", False):
            reward += 0.18
        if params.get("include_quality_score", False):
            reward += 0.18
        if params.get("include_recommendations", False):
            reward += 0.18

        missing_improved = dataset.isnull().sum().sum() < original.isnull().sum().sum()
        dups_improved = dataset.duplicated(subset=None, keep=False).sum() < original.duplicated(subset=None, keep=False).sum()
        if missing_improved:
            reward += 0.23
        if dups_improved:
            reward += 0.23
        if episode.delegated_work and any(v == "resolved" for v in episode.delegated_work.values()):
            reward += 0.1
        return min(1.0, reward)

    def _perform_delegation(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        actor = str(params.get("actor", "unknown")).strip().lower()
        objective = str(params.get("objective", "unspecified")).strip().lower()
        if not actor or actor == "unknown":
            return 0.0
        if actor not in {"sales_ops", "finance_bot", "support_lead", "compliance_officer"}:
            return 0.05
        self.current_episode.delegated_work[actor] = "queued"
        self.current_episode.actor_inbox.append(f"{actor}: accepted delegation for {objective or 'workflow task'}.")
        return 0.45

    def _perform_alert_resolution(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        actor = str(params.get("actor", "unknown")).strip().lower()
        if actor in self.current_episode.delegated_work:
            self.current_episode.delegated_work[actor] = "resolved"
            self.current_episode.actor_inbox.append(f"{actor}: escalation resolved.")
            return 0.7
        return 0.08

    def _perform_reconciliation(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        episode = self.current_episode
        dataset = episode.dataset
        key_col = params.get("join_key", "account_id")
        if key_col not in dataset.columns:
            return 0.0

        reward = 0.0
        if "crm_email" in dataset.columns and "invoice_status" in dataset.columns:
            inconsistent = (dataset["crm_email"].isna()) & (dataset["invoice_status"] == "overdue")
            if inconsistent.any():
                dataset.loc[inconsistent, "crm_email"] = "unknown@example.com"
                reward += 0.35
        if "ticket_status" in dataset.columns and "invoice_status" in dataset.columns:
            conflict = (dataset["ticket_status"] == "resolved") & (dataset["invoice_status"] == "overdue")
            if conflict.any():
                dataset.loc[conflict, "ticket_status"] = "in_progress"
                reward += 0.3
        if episode.drift_active and "compliance_tier" in dataset.columns and "region" in dataset.columns:
            needs_patch = (dataset["region"] == "EU") & (dataset["compliance_tier"] == "unknown")
            if needs_patch.any():
                dataset.loc[needs_patch, "compliance_tier"] = "strict"
                reward += 0.25
        return min(1.0, reward)

    def state(self) -> Dict[str, Any]:
        if not self.current_episode:
            return {"error": "No active episode"}
        episode = self.current_episode
        return {
            "dataset_shape": tuple(episode.dataset.shape),
            "missing_values_count": int(episode.dataset.isnull().sum().sum()),
            "duplicates_count": int(episode.dataset.duplicated(subset=None, keep=False).sum()),
            "columns": list(episode.dataset.columns),
            "step": episode.step_count,
            "task_id": episode.task_id,
            "dataset_name": episode.dataset_name,
            "seed": episode.seed,
            "actions": len(episode.actions_taken),
            "drift_active": episode.drift_active,
            "drift_notice": episode.drift_notice,
            "kpi_snapshot": dict(episode.kpis),
            "actor_messages": episode.actor_inbox[-5:],
            "delegated_work": dict(episode.delegated_work),
        }
