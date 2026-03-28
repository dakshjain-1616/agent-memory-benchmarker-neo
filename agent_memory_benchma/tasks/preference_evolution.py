"""Task 7 — Preference Evolution: preferences updated mid-conversation."""

from .base import Memory, Query, TaskSuite


class PreferenceEvolutionTask(TaskSuite):
    """Can the backend track the *latest* user preference when it has changed?"""

    name = "preference_evolution"
    description = "Store initial preferences then update each; query for latest values."

    @property
    def memories(self) -> list[Memory]:
        return [
            # Initial preferences
            Memory("User prefers vegetarian meals.", {"topic": "diet", "version": "v1"}),
            Memory("User prefers detailed, long-form explanations.", {"topic": "communication", "version": "v1"}),
            Memory("User's preferred tech stack is Python and Django.", {"topic": "tech", "version": "v1"}),
            Memory("User likes morning meetings before 10 AM.", {"topic": "meetings", "version": "v1"}),
            # Updates (contradictions)
            Memory(
                "UPDATE: User is now fully vegan, no longer just vegetarian.",
                {"topic": "diet", "version": "v2"},
            ),
            Memory(
                "UPDATE: User now prefers concise, bullet-point answers over long explanations.",
                {"topic": "communication", "version": "v2"},
            ),
            Memory(
                "UPDATE: User has switched to Go and Gin for their backend projects.",
                {"topic": "tech", "version": "v2"},
            ),
            Memory(
                "UPDATE: User now prefers afternoon meetings between 2–4 PM, not mornings.",
                {"topic": "meetings", "version": "v2"},
            ),
        ]

    @property
    def queries(self) -> list[Query]:
        return [
            Query(
                text="What is the user's current dietary preference?",
                expected_keywords=["vegan"],
                expected_phrase="vegan",
            ),
            Query(
                text="How does the user prefer to receive explanations?",
                expected_keywords=["concise", "bullet"],
                expected_phrase="concise",
            ),
            Query(
                text="What is the user's preferred tech stack?",
                expected_keywords=["go", "gin"],
                expected_phrase="Go",
            ),
            Query(
                text="When does the user prefer to have meetings?",
                expected_keywords=["afternoon", "2", "4 pm"],
                expected_phrase="afternoon",
            ),
            Query(
                text="Has the user's diet changed?",
                expected_keywords=["vegan", "vegetarian", "update", "changed"],
            ),
        ]
