import random
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from src.models import Observation, Action, Reward


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
    missing_values_identified: Dict[str, int] = field(default_factory=dict)
    duplicates_found: int = 0
    quality_issues: List[str] = field(default_factory=list)


class DataCleaningEnv:
    TASK_TEMPLATE_MAP = {
        "task_missing_values": "crm_contacts",
        "task_duplicate_handling": "billing_invoices",
        "task_complex_validation": "support_tickets",
    }

    def __init__(self, seed: int = DEFAULT_ENV_SEED):
        self.current_episode: Optional[EpisodeState] = None
        self.max_steps = 50
        self.default_seed = int(seed)
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._rng = random.Random(self.seed)
        self.dataset_templates = self._create_dataset_templates()

    def _create_dataset_templates(self) -> Dict[str, pd.DataFrame]:
        templates = {}

        rng = random.Random(self.seed)

        crm_data = pd.DataFrame({
            "contact_id": list(range(1, 121)),
            "account_id": [f"ACC{1000 + i}" for i in range(120)],
            "name": [f"Contact_{i}" if rng.random() > 0.13 else None for i in range(120)],
            "email": [f"contact{i}@example.com" if rng.random() > 0.17 else None for i in range(120)],
            "phone": [f"+1-202-555-{1000 + i}" if rng.random() > 0.19 else None for i in range(120)],
            "lead_source": [rng.choice(["website", "partner", "event", "outbound", None]) for _ in range(120)],
            "country": [rng.choice(["US", "UK", "CA", "DE", None]) for _ in range(120)],
        })
        crm_data = pd.concat([crm_data, crm_data.iloc[:12]], ignore_index=True)
        templates["crm_contacts"] = crm_data.reset_index(drop=True)

        billing_data = pd.DataFrame({
            "invoice_id": [f"INV{5000 + i}" for i in range(160)],
            "account_id": [f"ACC{1000 + (i % 90)}" for i in range(160)],
            "amount": [round(rng.uniform(100, 15000), 2) if rng.random() > 0.08 else None for _ in range(160)],
            "currency": [rng.choice(["USD", "EUR", "GBP", None]) for _ in range(160)],
            "status": [rng.choice(["paid", "pending", "overdue", None]) for _ in range(160)],
            "due_date": pd.date_range("2024-01-01", periods=160, freq="D").astype(str),
            "paid_date": [str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i + 5)) if rng.random() > 0.3 else None for i in range(160)],
        })
        billing_data = pd.concat([billing_data, billing_data.iloc[:18]], ignore_index=True)
        templates["billing_invoices"] = billing_data.reset_index(drop=True)

        support_data = pd.DataFrame({
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
        })
        support_data = pd.concat([support_data, support_data.iloc[:15]], ignore_index=True)
        templates["support_tickets"] = support_data.reset_index(drop=True)

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

    def reset(self, task_id: str = "task_missing_values", seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._set_seed(seed)

        template_name = self._select_template_name(task_id)
        selected_template = self.dataset_templates[template_name]
        dataset = selected_template.copy(deep=True)

        self.current_episode = EpisodeState(
            dataset=dataset,
            original_dataset=dataset.copy(),
            task_id=task_id,
            dataset_name=template_name,
            seed=self.seed,
            step_count=0
        )

        return self._get_observation()

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
            episode_progress=self._get_progress_summary()
        )

    def _describe_state(self) -> str:
        if not self.current_episode:
            return "No active episode"
        
        episode = self.current_episode
        rows, cols = episode.dataset.shape
        missing_count = episode.dataset.isnull().sum().sum()
        dup_count = episode.dataset.duplicated(subset=None, keep=False).sum()

        return (
            f"Dataset {episode.dataset_name} ({rows} rows, {cols} cols): "
            f"{missing_count} missing values, {dup_count} potential duplicates"
        )

    def _get_progress_summary(self) -> str:
        if not self.current_episode:
            return "No progress"
        
        episode = self.current_episode
        if not episode.actions_taken:
            return "No actions taken yet"
        
        return f"Completed {len(episode.actions_taken)} action(s): {', '.join([a.get('action_type', '?') for a in episode.actions_taken[-3:]])}"

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if not self.current_episode:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        
        episode = self.current_episode
        episode.step_count += 1
        
        episode.actions_taken.append({
            'action_type': action.action_type,
            'target_columns': action.target_columns,
            'parameters': action.parameters,
            'step': episode.step_count
        })
        
        reward, info = self._process_action(action)
        
        done = episode.step_count >= self.max_steps
        
        observation = self._get_observation()
        
        return observation, reward, done, info

    def _process_action(self, action: Action) -> Tuple[Reward, Dict[str, Any]]:
        if not self.current_episode:
            raise RuntimeError("No active episode")
        
        episode = self.current_episode
        components = {}
        messages = []
        
        if action.action_type == 'analyze':
            components['analysis'] = self._perform_analysis(action.target_columns)
            messages.append(f"Analyzed {len(action.target_columns)} columns")
        
        elif action.action_type == 'impute':
            impute_reward = self._perform_imputation(action.target_columns, action.parameters)
            components['imputation'] = impute_reward
            messages.append("Imputation applied")
        
        elif action.action_type == 'deduplicate':
            dedup_reward = self._perform_deduplication(action.parameters)
            components['deduplication'] = dedup_reward
            messages.append("Deduplication executed")
        
        elif action.action_type == 'validate':
            validation_reward = self._perform_validation(action.target_columns, action.parameters)
            components['validation'] = validation_reward
            messages.append("Validation performed")
        
        elif action.action_type == 'report_findings':
            report_reward = self._generate_report(action.parameters)
            components['reporting'] = report_reward
            messages.append("Report generated")
        
        else:
            components['invalid_action'] = 0.0
            messages.append(f"Unknown action type: {action.action_type}")
        
        if not components:
            total_reward = 0.0
        else:
            total_reward = sum(components.values()) / len(components) if components else 0.0
        
        total_reward = min(1.0, max(0.0, total_reward))
        
        info = {
            'action_type': action.action_type,
            'reasoning': action.reasoning,
            'components': components,
            'messages': messages
        }
        
        return Reward(
            value=total_reward,
            components=components,
            message="; ".join(messages)
        ), info

    def _perform_analysis(self, columns: List[str]) -> float:
        if not self.current_episode:
            return 0.0
        
        dataset = self.current_episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        
        if not valid_cols:
            return 0.0
        
        reward = 0.0
        for col in valid_cols:
            missing_pct = dataset[col].isnull().sum() / len(dataset)
            if missing_pct > 0:
                reward += 0.3
            
            if dataset[col].dtype == 'object':
                reward += 0.1
        
        return min(1.0, reward / len(valid_cols))

    def _perform_imputation(self, columns: List[str], params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        episode = self.current_episode
        dataset = episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        
        if not valid_cols:
            return 0.0
        
        reward = 0.0
        method = params.get('method', 'mean')
        
        for col in valid_cols:
            if dataset[col].isnull().sum() == 0:
                reward += 0.1
            elif method in ['mean', 'median'] and pd.api.types.is_numeric_dtype(dataset[col]):
                if method == 'mean':
                    value = dataset[col].mean()
                else:
                    value = dataset[col].median()
                
                if pd.notna(value):
                    dataset[col] = dataset[col].fillna(value)
                    missing_after = dataset[col].isnull().sum()
                    reward += 0.4 if missing_after == 0 else 0.2
            elif method == 'forward_fill':
                dataset[col] = dataset[col].ffill()
                reward += 0.35
        
        return min(1.0, reward / len(valid_cols)) if valid_cols else 0.0

    def _perform_deduplication(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        episode = self.current_episode
        dataset = episode.dataset
        
        dup_count_before = dataset.duplicated(subset=None, keep=False).sum()
        
        if dup_count_before == 0:
            return 0.5
        
        subset = params.get('subset', None)
        keep = params.get('keep', 'first')
        
        if subset and all(c in dataset.columns for c in subset):
            dataset.drop_duplicates(subset=subset, keep=keep, inplace=True)
        else:
            dataset.drop_duplicates(keep=keep, inplace=True)
        
        dup_count_after = dataset.duplicated(subset=None, keep=False).sum()
        
        if dup_count_after == 0:
            return 0.9
        else:
            return max(0.4, 1.0 - (dup_count_after / dup_count_before))

    def _perform_validation(self, columns: List[str], params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        dataset = self.current_episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        
        if not valid_cols:
            return 0.0
        
        reward = 0.0
        
        for col in valid_cols:
            validation_type = params.get(f'{col}_type', 'exists')
            
            if validation_type == 'exists':
                if dataset[col].isnull().sum() == 0:
                    reward += 0.5
                else:
                    reward += 0.2
            
            elif validation_type == 'numeric':
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    reward += 0.5
                else:
                    reward += 0.1
            
            elif validation_type == 'range':
                if not pd.api.types.is_numeric_dtype(dataset[col]):
                    reward += 0.05
                    continue
                min_val = params.get(f'{col}_min', dataset[col].min())
                max_val = params.get(f'{col}_max', dataset[col].max())
                
                in_range = ((dataset[col] >= min_val) & (dataset[col] <= max_val)).sum()
                reward += (in_range / len(dataset)) * 0.5
        
        return min(1.0, reward / len(valid_cols)) if valid_cols else 0.0

    def _generate_report(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        episode = self.current_episode
        dataset = episode.dataset
        original = episode.original_dataset
        
        reward = 0.0
        
        if params.get('include_summary', False):
            reward += 0.2
        
        if params.get('include_quality_score', False):
            reward += 0.2
        
        if params.get('include_recommendations', False):
            reward += 0.2
        
        rows_original = len(original)
        rows_cleaned = len(dataset)
        cols_original = len(original.columns)
        cols_cleaned = len(dataset.columns)
        
        if rows_cleaned < rows_original:
            reward += 0.15
        
        if dataset.isnull().sum().sum() < original.isnull().sum().sum():
            reward += 0.25
        
        return min(1.0, reward)

    def state(self) -> Dict[str, Any]:
        if not self.current_episode:
            return {'error': 'No active episode'}
        
        episode = self.current_episode
        
        return {
            'dataset_shape': tuple(episode.dataset.shape),
            'missing_values_count': int(episode.dataset.isnull().sum().sum()),
            'duplicates_count': int(episode.dataset.duplicated(subset=None, keep=False).sum()),
            'columns': list(episode.dataset.columns),
            'step': episode.step_count,
            'task_id': episode.task_id,
            'dataset_name': episode.dataset_name,
            'seed': episode.seed,
            'actions': len(episode.actions_taken)
        }
