"""Task 2 — Temporal Ordering: recall events in correct chronological order."""

from .base import Memory, Query, TaskSuite


class TemporalOrderingTask(TaskSuite):
    """Can the backend retrieve which event happened first / last?"""

    name = "temporal_ordering"
    description = "Store a timeline of 7 historical events; query order relationships."

    @property
    def memories(self) -> list[Memory]:
        return [
            Memory("1969-07-20: Apollo 11 landed on the Moon.", {"date": "1969-07-20"}),
            Memory("1989-11-09: The Berlin Wall fell.", {"date": "1989-11-09"}),
            Memory("1991-08-06: Tim Berners-Lee published the first website.", {"date": "1991-08-06"}),
            Memory("2001-09-11: Terrorist attacks struck New York and Washington.", {"date": "2001-09-11"}),
            Memory("2008-09-15: Lehman Brothers filed for bankruptcy, triggering the financial crisis.", {"date": "2008-09-15"}),
            Memory("2020-03-11: The WHO declared COVID-19 a global pandemic.", {"date": "2020-03-11"}),
            Memory("1945-09-02: Japan signed the surrender ending World War II.", {"date": "1945-09-02"}),
        ]

    @property
    def queries(self) -> list[Query]:
        return [
            Query(
                text="When did Apollo 11 land on the Moon?",
                expected_keywords=["1969", "apollo"],
                expected_phrase="1969",
            ),
            Query(
                text="What happened on 1989-11-09?",
                expected_keywords=["berlin", "wall"],
                expected_phrase="Berlin Wall",
            ),
            Query(
                text="When was the first website published?",
                expected_keywords=["1991", "berners-lee", "berners"],
                expected_phrase="1991",
            ),
            Query(
                text="When did the WHO declare COVID-19 a pandemic?",
                expected_keywords=["2020", "pandemic", "covid"],
                expected_phrase="2020",
            ),
            Query(
                text="What event ended World War II?",
                expected_keywords=["japan", "surrender", "1945"],
                expected_phrase="1945",
            ),
        ]
