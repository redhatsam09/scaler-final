from src.environment import DataCleaningEnv
from src.graders import MissingValuesGrader, DuplicateHandlingGrader, ComplexValidationGrader
from inference import local_policy_action


TASKS = [
    ("task_missing_values", MissingValuesGrader),
    ("task_duplicate_handling", DuplicateHandlingGrader),
    ("task_complex_validation", ComplexValidationGrader),
]


def _state_line(state: dict) -> str:
    return (
        f"dataset={state['dataset_name']} shape={state['dataset_shape']} "
        f"missing={state['missing_values_count']} duplicates={state['duplicates_count']} "
        f"actions={state['actions']}"
    )


def run_demo() -> None:
    print("WORLD MODELING VALIDATION DEMO")
    print("Hidden world state: enterprise business dataset")
    print("Visible observation: schema, missing-value counts, shape, progress summary")
    print("Agent actions: analyze -> clean -> validate -> report")
    print()

    for task_index, (task_id, grader_class) in enumerate(TASKS):
        env = DataCleaningEnv()
        observation = env.reset(task_id=task_id, seed=2026 + task_index)
        print(f"TASK {task_id}")
        print(f"reset: {_state_line(env.state())}")

        for step in range(1, 6):
            before = env.state()
            action = local_policy_action(task_id, observation, step)
            observation, reward, done, _info = env.step(action)
            after = env.state()
            score = grader_class.grade(env.current_episode)
            print(
                f"step={step} action={action.action_type} reward={reward.value:.3f} "
                f"missing {before['missing_values_count']}->{after['missing_values_count']} "
                f"duplicates {before['duplicates_count']}->{after['duplicates_count']} "
                f"grade={score:.3f}"
            )
            if done:
                break

        print(f"final: {_state_line(env.state())} score={grader_class.grade(env.current_episode):.3f}")
        print()


if __name__ == "__main__":
    run_demo()
