from check_pc import assess


def test_verdicts():
    assert assess([], 32, 100)[0] == "NOT PRACTICAL"
    assert assess([("NVIDIA GeForce RTX 4090", 24, "x")], 64, 100)[0] == "EXCELLENT"
    assert assess([("NVIDIA GeForce RTX 3060", 12, "x")], 32, 100)[0] == "GOOD"
    assert assess([("NVIDIA GeForce RTX 3070", 8, "x")], 32, 100)[0] == "POSSIBLE BUT SLOW"
    assert assess([("NVIDIA GeForce RTX 3060", 12, "x")], 8, 100)[0] == "NOT PRACTICAL"
    assert assess([("NVIDIA GeForce RTX 3060", 12, "x")], 32, 10)[0] == "GOOD (after freeing disk space)"
