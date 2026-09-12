from triage.features.dataset import combine_title_body


def test_combine_title_body_joins_title_and_body():
    assert combine_title_body("App crashes", "Stack trace here") == "App crashes\nStack trace here"


def test_combine_title_body_handles_missing_body():
    assert combine_title_body("App crashes", None) == "App crashes"


def test_combine_title_body_strips_surrounding_whitespace():
    assert combine_title_body("  App crashes  ", "  ") == "App crashes"
