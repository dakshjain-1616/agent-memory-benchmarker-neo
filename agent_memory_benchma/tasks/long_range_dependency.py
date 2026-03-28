"""Task 5 — Long-range Dependency: answer requires chaining multiple memories."""

from .base import Memory, Query, TaskSuite


class LongRangeDependencyTask(TaskSuite):
    """Can the backend surface all relevant chunks for multi-hop questions?

    Each query requires information stored across at least 2 separate memories.
    """

    name = "long_range_dependency"
    description = "Store 10 memories; queries require chaining 2+ memories to answer."

    @property
    def memories(self) -> list[Memory]:
        return [
            # Project chain
            Memory("Project Atlas is led by Sarah Chen.", {"topic": "atlas"}),
            Memory("Sarah Chen reports directly to the CTO, Marcus Webb.", {"topic": "atlas"}),
            Memory("Marcus Webb joined Acme Corp in January 2020.", {"topic": "acme"}),
            # Supply chain
            Memory("Widget X is manufactured by Factory Delta in Shenzhen.", {"topic": "widget"}),
            Memory("Factory Delta has ISO 9001 certification and ships within 7 days.", {"topic": "widget"}),
            Memory("Widget X requires three components: sensor A, motor B, and casing C.", {"topic": "widget"}),
            # Research chain
            Memory("The Prometheus study enrolled 1,200 patients across 15 sites.", {"topic": "prometheus"}),
            Memory("The Prometheus study was funded by the Hartwell Foundation.", {"topic": "prometheus"}),
            Memory("The Hartwell Foundation's annual grant budget is $40 million.", {"topic": "hartwell"}),
            Memory("The Prometheus study concluded that Treatment P reduces risk by 32%.", {"topic": "prometheus"}),
        ]

    @property
    def queries(self) -> list[Query]:
        return [
            Query(
                text="Who does the leader of Project Atlas report to?",
                expected_keywords=["marcus", "webb", "cto"],
                expected_phrase="Marcus Webb",
            ),
            Query(
                text="When did the CTO of Acme Corp join the company?",
                expected_keywords=["2020", "january"],
                expected_phrase="2020",
            ),
            Query(
                text="Where is Widget X manufactured, and how long does shipping take?",
                expected_keywords=["shenzhen", "7 days", "delta"],
                expected_phrase="Shenzhen",
            ),
            Query(
                text="Who funded the Prometheus study?",
                expected_keywords=["hartwell", "foundation"],
                expected_phrase="Hartwell Foundation",
            ),
            Query(
                text="What was the outcome of the Prometheus study?",
                expected_keywords=["32", "treatment", "reduces", "risk"],
                expected_phrase="32%",
            ),
            Query(
                text="What is the annual grant budget of the foundation that funded Prometheus?",
                expected_keywords=["40 million", "40,000,000", "$40"],
                expected_phrase="$40 million",
            ),
        ]
