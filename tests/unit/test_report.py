from atf_eval.report import render_results_table


def test_empty_conversations():
    assert render_results_table([]) == "(no conversations)"


def test_renders_score_and_na_correctly():
    table = render_results_table(
        [
            {"conversation_id": "c1", "atf": 0.749, "metric_coverage": 0.9, "nts": 0.946, "sts": 0.8, "tis": None, "os": 0.0},
        ]
    )
    lines = table.splitlines()
    assert lines[0].split() == ["Conversation", "ATF", "Coverage", "NTS", "STS", "TIS", "OS"]
    assert "c1" in lines[2]
    assert "0.749" in lines[2]
    assert "90.0%" in lines[2]
    assert "N/A" in lines[2]


def test_column_width_adapts_to_longest_conversation_id():
    table = render_results_table(
        [{"conversation_id": "a_very_long_conversation_id", "atf": 1.0, "metric_coverage": 1.0,
          "nts": 1.0, "sts": 1.0, "tis": 1.0, "os": 1.0}]
    )
    assert "a_very_long_conversation_id" in table.splitlines()[2]
