import pytest

from atf_eval.dataset import load_dataset
from atf_eval.loader import load_adapter
from atf_eval.scaffold import write_scaffold


def test_writes_all_three_files(tmp_path):
    written = write_scaffold(tmp_path)
    names = {p.name for p in written}
    assert names == {"adapter.py", "golden_dataset.jsonl", "README.md"}
    for p in written:
        assert p.exists()


def test_refuses_to_overwrite_without_force(tmp_path):
    write_scaffold(tmp_path)
    with pytest.raises(FileExistsError):
        write_scaffold(tmp_path)


def test_force_overwrites(tmp_path):
    write_scaffold(tmp_path)
    write_scaffold(tmp_path, force=True)  # should not raise


def test_golden_dataset_is_valid_and_loadable(tmp_path):
    write_scaffold(tmp_path)
    turns = load_dataset(str(tmp_path / "golden_dataset.jsonl"))
    assert len(turns) == 2
    assert turns[0].conversation_id == turns[1].conversation_id


def test_adapter_is_importable_and_matches_golden_dataset(tmp_path):
    write_scaffold(tmp_path)
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        adapter = load_adapter("adapter:MyAgentAdapter")
        turns = load_dataset(str(tmp_path / "golden_dataset.jsonl"))

        history = []
        for t in turns:
            observed = adapter.run_turn(t.conversation_id, t.turn_id, t.user_input, history)
            history.append(observed)
            expected = t.to_normalized()
            assert observed.nodes[0].node_id == expected.nodes[0].node_id
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("adapter", None)
