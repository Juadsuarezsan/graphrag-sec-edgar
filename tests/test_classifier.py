from src.router.classifier import classify


def test_lookup():
    assert classify("Who is the CEO of Apple?") == "lookup"


def test_multi_hop():
    assert classify("Which companies are supplied by TSMC?") == "multi_hop"


def test_aggregation():
    assert classify("How many S&P 500 companies?") == "aggregation"


def test_hybrid_when_unclear():
    assert classify("Tell me something about Apple") == "hybrid"
