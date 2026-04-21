import pandas as pd
from typing import Dict, Any
from src.environment import DataCleaningEnv, EpisodeState


MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99


def _strict_task_score(value: float) -> float:
    if value <= MIN_TASK_SCORE:
        return MIN_TASK_SCORE
    if value >= MAX_TASK_SCORE:
        return MAX_TASK_SCORE
    return float(value)


class MissingValuesGrader:
    
    @staticmethod
    def grade(episode_state: EpisodeState) -> float:
        dataset = episode_state.dataset
        original_dataset = episode_state.original_dataset
        
        original_missing = original_dataset.isnull().sum().sum()
        current_missing = dataset.isnull().sum().sum()
        
        if original_missing == 0:
            return _strict_task_score(0.5)
        
        reduction = (original_missing - current_missing) / original_missing
        
        analysis_performed = any(
            a['action_type'] == 'analyze' 
            for a in episode_state.actions_taken
        )
        
        imputation_performed = any(
            a['action_type'] == 'impute' 
            for a in episode_state.actions_taken
        )
        
        base_score = max(0.0, reduction)
        
        if analysis_performed:
            base_score += 0.15
        
        if imputation_performed:
            base_score += 0.25
        
        if current_missing == 0:
            base_score = min(1.0, base_score + 0.2)
        
        return _strict_task_score(min(1.0, base_score))


class DuplicateHandlingGrader:
    
    @staticmethod
    def grade(episode_state: EpisodeState) -> float:
        dataset = episode_state.dataset
        original_dataset = episode_state.original_dataset
        
        original_dups = original_dataset.duplicated(subset=None, keep=False).sum()
        current_dups = dataset.duplicated(subset=None, keep=False).sum()
        
        if original_dups == 0:
            return _strict_task_score(0.4)
        
        dedup_performed = any(
            a['action_type'] == 'deduplicate' 
            for a in episode_state.actions_taken
        )
        
        if not dedup_performed:
            return _strict_task_score(0.2)
        
        reduction = (original_dups - current_dups) / original_dups
        base_score = max(0.0, reduction)
        
        if current_dups == 0:
            base_score = min(1.0, base_score + 0.3)
        
        validation_performed = any(
            a['action_type'] == 'validate' 
            for a in episode_state.actions_taken
        )
        
        if validation_performed:
            base_score = min(1.0, base_score + 0.1)
        
        return _strict_task_score(min(1.0, base_score))


class ComplexValidationGrader:
    
    @staticmethod
    def grade(episode_state: EpisodeState) -> float:
        dataset = episode_state.dataset
        original_dataset = episode_state.original_dataset
        
        score = 0.0
        
        missing_pct_original = (original_dataset.isnull().sum().sum() / 
                               (len(original_dataset) * len(original_dataset.columns)))
        missing_pct_current = (dataset.isnull().sum().sum() / 
                              (len(dataset) * len(dataset.columns)))
        
        missing_improvement = max(0.0, (missing_pct_original - missing_pct_current) / missing_pct_original if missing_pct_original > 0 else 0.5)
        score += missing_improvement * 0.25
        
        dup_pct_original = original_dataset.duplicated(subset=None, keep=False).sum() / len(original_dataset)
        dup_pct_current = dataset.duplicated(subset=None, keep=False).sum() / len(dataset)
        
        dup_improvement = max(0.0, (dup_pct_original - dup_pct_current) / dup_pct_original if dup_pct_original > 0 else 0.5)
        score += dup_improvement * 0.20
        
        actions_count = len(episode_state.actions_taken)
        action_diversity = len(set(a['action_type'] for a in episode_state.actions_taken))
        
        if actions_count >= 5:
            score += 0.15
        elif actions_count >= 3:
            score += 0.10
        else:
            score += 0.05
        
        if action_diversity >= 3:
            score += 0.10
        elif action_diversity >= 2:
            score += 0.05
        
        validation_performed = any(
            a['action_type'] == 'validate' 
            for a in episode_state.actions_taken
        )
        
        if validation_performed:
            score += 0.15
        
        analysis_performed = any(
            a['action_type'] == 'analyze' 
            for a in episode_state.actions_taken
        )
        
        if analysis_performed:
            score += 0.10
        
        report_generated = any(
            a['action_type'] == 'report_findings' 
            for a in episode_state.actions_taken
        )
        
        if report_generated:
            score += 0.05
        
        return _strict_task_score(min(1.0, score))
