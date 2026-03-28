"""Task 6 — Multi-Session: facts introduced in earlier sessions recalled later."""

from .base import Memory, Query, TaskSuite


class MultiSessionTask(TaskSuite):
    """Can the backend retain information across simulated conversation sessions?"""

    name = "multi_session"
    description = "Store memories across 3 simulated sessions; query cross-session facts."

    @property
    def memories(self) -> list[Memory]:
        return [
            # Session 1
            Memory("[Session 1] Jordan Lee founded TimeFlow in 2022.", {"session": 1}),
            Memory("[Session 1] TimeFlow's co-founder is Elena Vasquez.", {"session": 1}),
            Memory("[Session 1] TimeFlow plans a public beta in Q3.", {"session": 1}),
            # Session 2
            Memory("[Session 2] TimeFlow raised $1.2 million in seed funding.", {"session": 2}),
            Memory("[Session 2] Two new engineers joined: Ravi Patel and Diane Wu.", {"session": 2}),
            Memory("[Session 2] TimeFlow signed a pilot agreement with Globex Corp.", {"session": 2}),
            # Session 3
            Memory("[Session 3] The Q3 beta shipped on schedule and received positive reviews.", {"session": 3}),
            Memory("[Session 3] Globex Corp extended the pilot to a full contract worth $200,000.", {"session": 3}),
            Memory("[Session 3] TimeFlow is now hiring a Head of Sales.", {"session": 3}),
        ]

    @property
    def queries(self) -> list[Query]:
        return [
            Query(
                text="Who founded TimeFlow?",
                expected_keywords=["jordan", "lee"],
                expected_phrase="Jordan Lee",
            ),
            Query(
                text="Who is the co-founder of TimeFlow?",
                expected_keywords=["elena", "vasquez"],
                expected_phrase="Elena Vasquez",
            ),
            Query(
                text="How much seed funding did TimeFlow raise?",
                expected_keywords=["1.2 million", "1,200,000", "$1.2"],
                expected_phrase="$1.2 million",
            ),
            Query(
                text="Which company signed a pilot agreement with TimeFlow?",
                expected_keywords=["globex"],
                expected_phrase="Globex Corp",
            ),
            Query(
                text="What was the value of Globex Corp's full contract?",
                expected_keywords=["200,000", "$200"],
                expected_phrase="$200,000",
            ),
            Query(
                text="Which new engineers joined TimeFlow in session 2?",
                expected_keywords=["ravi", "diane"],
                expected_phrase="Ravi",
            ),
        ]
