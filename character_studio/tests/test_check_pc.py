from check_pc import assess


def nv(name, vram):
    return [(name, vram, "NVIDIA")]


def test_nvidia_verdicts():
    assert assess([], 32, 100)[0] == "NOT PRACTICAL"
    assert assess(nv("NVIDIA GeForce RTX 4090", 24), 64, 100)[0] == "EXCELLENT"
    assert assess(nv("NVIDIA GeForce RTX 3060", 12), 32, 100)[0] == "GOOD"
    assert assess(nv("NVIDIA GeForce RTX 3070", 8), 32, 100)[0] == "POSSIBLE BUT SLOW"
    assert assess(nv("NVIDIA GeForce GTX 1070", 8), 16, 1300)[0] == "POSSIBLE BUT SLOW"
    assert assess(nv("NVIDIA GeForce RTX 3060", 12), 8, 100)[0] == "NOT PRACTICAL"
    assert assess(nv("NVIDIA GeForce RTX 3060", 12), 32, 10)[0] == "GOOD (after freeing disk space)"


def test_amd_verdicts():
    verdict, lines = assess([("AMD Radeon RX 9060 XT", 16, "AMD")], 32, 500)
    assert verdict == "GOOD" and any("20-60 seconds" in line for line in lines)
    assert assess([("AMD Radeon RX 7800 XT", 16, "AMD")], 32, 500)[0] == "GOOD"
    assert assess([("AMD Radeon RX 9060 XT", 8, "AMD")], 32, 500)[0] == "POSSIBLE BUT SLOW"
    verdict, lines = assess([("AMD Radeon RX 6800", 16, "AMD")], 32, 500)
    assert verdict == "NOT PRACTICAL" and "too old" in lines[0]
    # a Ryzen's built-in graphics next to a supported card must not hide the card
    assert assess([("AMD Radeon(TM) Graphics", 0.5, "AMD"),
                   ("AMD Radeon RX 9060 XT", 16, "AMD")], 32, 500)[0] == "GOOD"


def test_amd_vram_read_from_registry(monkeypatch):
    import sys
    import types

    import check_pc
    entries = {"0000": {"DriverDesc": "AMD Radeon(TM) Graphics",
                        "HardwareInformation.qwMemorySize": 512 * 1024 ** 2},
               "0001": {"DriverDesc": "AMD Radeon RX 9060 XT",
                        "HardwareInformation.qwMemorySize": (16 * 1024 ** 3).to_bytes(8, "little")},
               "Properties": {}}
    order = list(entries)

    class Key:
        def __init__(self, sub=None):
            self.sub = sub

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def open_key(base, sub):
        return Key(None if isinstance(base, int) else sub)

    def enum_key(key, i):
        if i >= len(order):
            raise OSError
        return order[i]

    def query(key, value):
        try:
            return entries[key.sub][value], 0
        except KeyError:
            raise OSError

    fake = types.SimpleNamespace(HKEY_LOCAL_MACHINE=0, OpenKey=open_key, EnumKey=enum_key,
                                 QueryValueEx=query)
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(check_pc, "os", types.SimpleNamespace(name="nt"))
    found = dict(check_pc.amd_gpus())
    assert found["AMD Radeon RX 9060 XT"] == 16
    assert found["AMD Radeon(TM) Graphics"] == 0.5
