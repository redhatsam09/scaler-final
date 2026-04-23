from typing import Dict, List


DEFAULT_REPORT_PARAMS = {
    "include_summary": True,
    "include_quality_score": True,
    "include_recommendations": True,
}

EXTENDED_REPORT_PARAMS = {
    "include_summary": True,
    "include_quality_score": True,
    "include_recommendations": True,
    "include_actor_tradeoffs": True,
    "include_budget_analysis": True,
}

ACTOR_ACTIONS = {"delegate", "resolve_alert", "oversight_review", "inspect_actor"}

ACTION_SPACE: Dict[str, List[Dict]] = {
    "task_missing_values": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "impute", "parameters": {"method": "forward_fill"}},
        {"action_type": "impute", "parameters": {"method": "mean"}},
        {"action_type": "deduplicate", "parameters": {"keep": "first"}},
        {"action_type": "validate", "parameters": {}},
        {"action_type": "report_findings", "parameters": dict(DEFAULT_REPORT_PARAMS)},
    ],
    "task_duplicate_handling": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "deduplicate", "parameters": {"keep": "first"}},
        {"action_type": "deduplicate", "parameters": {"keep": "last"}},
        {"action_type": "validate", "parameters": {}},
        {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
        {"action_type": "report_findings", "parameters": dict(DEFAULT_REPORT_PARAMS)},
    ],
    "task_complex_validation": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "impute", "parameters": {"method": "forward_fill"}},
        {"action_type": "deduplicate", "parameters": {"keep": "first"}},
        {"action_type": "validate", "parameters": {}},
        {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
        {"action_type": "report_findings", "parameters": dict(DEFAULT_REPORT_PARAMS)},
    ],
    "task_enterprise_orchestration": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "inspect_actor", "parameters": {"actor": "finance_bot"}},
        {"action_type": "inspect_actor", "parameters": {"actor": "analytics_assistant"}},
        {"action_type": "delegate", "parameters": {"actor": "finance_bot", "objective": "invoice cleanup"}},
        {"action_type": "delegate", "parameters": {"actor": "support_lead", "objective": "critical ticket triage"}},
        {"action_type": "delegate", "parameters": {"actor": "sales_ops", "objective": "protect conversion"}},
        {"action_type": "resolve_alert", "parameters": {"actor": "finance_bot"}},
        {"action_type": "resolve_alert", "parameters": {"actor": "support_lead"}},
        {"action_type": "oversight_review", "parameters": {"actor": "analytics_assistant", "explain": True}},
        {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
        {"action_type": "request_policy_clarification", "parameters": {}},
        {
            "action_type": "validate",
            "parameters": {
                "compliance_tier_type": "categorical_nonempty",
                "ticket_priority_type": "categorical_nonempty",
            },
        },
        {"action_type": "report_findings", "parameters": dict(EXTENDED_REPORT_PARAMS)},
    ],
}
