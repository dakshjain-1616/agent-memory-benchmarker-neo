"""Task 4 — Contradiction Detection: store conflicting info, retrieve latest."""

from .base import Memory, Query, TaskSuite


class ContradictionDetectionTask(TaskSuite):
    """Can the backend surface the most recent / corrected version of a fact?

    This tests whether backends can handle updates and contradictions without
    losing the corrected value.
    """

    name = "contradiction_detection"
    description = (
        "Store initial facts followed by corrections; query for the correct value."
    )

    @property
    def memories(self) -> list[Memory]:
        return [
            # Initial assertions
            Memory("The project deadline is March 15.", {"version": "v1", "topic": "deadline"}),
            Memory("The budget for the campaign is $50,000.", {"version": "v1", "topic": "budget"}),
            Memory("The meeting is scheduled for Tuesday at 9 AM.", {"version": "v1", "topic": "meeting"}),
            Memory("The recommended dosage is 200 mg twice daily.", {"version": "v1", "topic": "dosage"}),
            # Corrections / updates (stored later)
            Memory(
                "CORRECTION: The project deadline has been moved to April 1, not March 15.",
                {"version": "v2", "topic": "deadline"},
            ),
            Memory(
                "UPDATE: The campaign budget has been increased to $75,000.",
                {"version": "v2", "topic": "budget"},
            ),
            Memory(
                "The meeting has been rescheduled to Wednesday at 2 PM. Tuesday at 9 AM is cancelled.",
                {"version": "v2", "topic": "meeting"},
            ),
            Memory(
                "Dosage update: the correct dosage is 100 mg three times daily, not 200 mg twice.",
                {"version": "v2", "topic": "dosage"},
            ),
        ]

    @property
    def queries(self) -> list[Query]:
        return [
            Query(
                text="What is the current project deadline?",
                expected_keywords=["april", "1", "correction"],
                expected_phrase="April 1",
            ),
            Query(
                text="What is the campaign budget?",
                expected_keywords=["75,000", "75000", "increased"],
                expected_phrase="75,000",
            ),
            Query(
                text="When is the meeting?",
                expected_keywords=["wednesday", "2 pm", "rescheduled"],
                expected_phrase="Wednesday",
            ),
            Query(
                text="What is the recommended dosage?",
                expected_keywords=["100 mg", "100mg", "three", "three times"],
                expected_phrase="100 mg",
            ),
        ]
