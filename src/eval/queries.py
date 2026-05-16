"""Eval set covering 4 query kinds (lookup / multi_hop / aggregation / hybrid)."""

QUERIES: list[dict] = [
    {"id": "lk-01", "kind": "lookup", "question": "Who is the CEO of Apple?",
     "must_cite": ["tim_cook", "apple"]},
    {"id": "lk-02", "kind": "lookup", "question": "What is NVIDIA's sector?",
     "must_cite": ["nvidia"]},
    {"id": "lk-03", "kind": "lookup", "question": "Where is TSMC based?",
     "must_cite": ["tsmc"]},

    {"id": "mh-01", "kind": "multi_hop",
     "question": "Which companies are supplied by TSMC?",
     "must_cite": ["tsmc", "apple", "nvidia"]},
    {"id": "mh-02", "kind": "multi_hop",
     "question": "What competitors of Apple also depend on the same suppliers?",
     "must_cite": ["apple", "samsung"]},
    {"id": "mh-03", "kind": "multi_hop",
     "question": "Companies connected to risk_china_exposure via supplier?",
     "must_cite": ["risk_china_exposure"]},

    {"id": "agg-01", "kind": "aggregation",
     "question": "How many S&P 500 companies are in the graph?", "expected_count_min": 10},
    {"id": "agg-02", "kind": "aggregation",
     "question": "How many people are CEOs in the graph?", "expected_count_min": 8},
    {"id": "agg-03", "kind": "aggregation",
     "question": "Count of risks?", "expected_count_min": 4},
]
