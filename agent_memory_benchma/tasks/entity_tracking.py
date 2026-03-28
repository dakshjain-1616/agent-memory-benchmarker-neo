"""Task 3 — Entity Tracking: follow multiple attributes of named entities."""

from .base import Memory, Query, TaskSuite


class EntityTrackingTask(TaskSuite):
    """Can the backend track changing attributes of distinct entities?"""

    name = "entity_tracking"
    description = "Store 10 facts about 4 entities; query per-entity attributes."

    @property
    def memories(self) -> list[Memory]:
        return [
            Memory("Bob is 34 years old and works as a software engineer.", {"entity": "Bob"}),
            Memory("Bob lives in Berlin and speaks German and English.", {"entity": "Bob"}),
            Memory("Carol is 29 years old and is a data scientist.", {"entity": "Carol"}),
            Memory("Carol's favourite programming language is Python.", {"entity": "Carol"}),
            Memory("David is the CEO of Acme Corp and has been in the role since 2018.", {"entity": "David"}),
            Memory("David owns three properties: one in London, one in Tokyo, one in Dubai.", {"entity": "David"}),
            Memory("Eve is a PhD student researching quantum computing at MIT.", {"entity": "Eve"}),
            Memory("Eve's advisor is Professor Hart and her thesis is due in 2026.", {"entity": "Eve"}),
            Memory("Acme Corp was founded in 2005 and has 500 employees.", {"entity": "Acme Corp"}),
            Memory("Acme Corp's main product is an AI-powered analytics platform.", {"entity": "Acme Corp"}),
        ]

    @property
    def queries(self) -> list[Query]:
        return [
            Query(
                text="How old is Bob and what is his job?",
                expected_keywords=["34", "engineer", "software"],
                expected_phrase="software engineer",
            ),
            Query(
                text="What is Carol's favourite programming language?",
                expected_keywords=["python"],
                expected_phrase="Python",
            ),
            Query(
                text="When did David become CEO?",
                expected_keywords=["2018", "ceo"],
                expected_phrase="2018",
            ),
            Query(
                text="Where is Eve doing her PhD?",
                expected_keywords=["mit", "quantum"],
                expected_phrase="MIT",
            ),
            Query(
                text="How many employees does Acme Corp have?",
                expected_keywords=["500", "employees"],
                expected_phrase="500",
            ),
            Query(
                text="What cities does David own properties in?",
                expected_keywords=["london", "tokyo", "dubai"],
                expected_phrase="London",
            ),
        ]
