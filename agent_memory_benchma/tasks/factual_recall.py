"""Task 1 — Factual Recall: store discrete facts, retrieve on demand."""

from .base import Memory, Query, TaskSuite


class FactualRecallTask(TaskSuite):
    """Can the backend return the exact fact asked for?"""

    name = "factual_recall"
    description = "Store 8 unrelated facts; query each individually."

    @property
    def memories(self) -> list[Memory]:
        return [
            Memory("Alice's favourite colour is cerulean blue."),
            Memory("The Eiffel Tower was completed in 1889."),
            Memory("Python was created by Guido van Rossum."),
            Memory("The speed of light is approximately 299,792 kilometres per second."),
            Memory("Mount Everest is 8,849 metres above sea level."),
            Memory("Shakespeare wrote 37 plays and 154 sonnets."),
            Memory("The chemical symbol for gold is Au."),
            Memory("Water boils at 100 degrees Celsius at standard pressure."),
        ]

    @property
    def queries(self) -> list[Query]:
        return [
            Query(
                text="What is Alice's favourite colour?",
                expected_keywords=["cerulean", "blue"],
                expected_phrase="cerulean blue",
            ),
            Query(
                text="When was the Eiffel Tower completed?",
                expected_keywords=["1889"],
                expected_phrase="1889",
            ),
            Query(
                text="Who created Python?",
                expected_keywords=["guido", "rossum"],
                expected_phrase="Guido van Rossum",
            ),
            Query(
                text="What is the speed of light?",
                expected_keywords=["299", "792"],
                expected_phrase="299,792",
            ),
            Query(
                text="How tall is Mount Everest?",
                expected_keywords=["8,849", "8849", "metres"],
                expected_phrase="8,849",
            ),
            Query(
                text="What is the chemical symbol for gold?",
                expected_keywords=["au"],
                expected_phrase="Au",
            ),
            Query(
                text="At what temperature does water boil?",
                expected_keywords=["100", "celsius"],
                expected_phrase="100 degrees Celsius",
            ),
        ]
