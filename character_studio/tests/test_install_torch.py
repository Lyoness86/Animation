import sys

import install_torch


def fake(monkeypatch, nvidia=(), amd=(), py=(3, 12)):
    calls = []
    monkeypatch.setattr(install_torch, "nvidia_gpus", lambda: list(nvidia))
    monkeypatch.setattr(install_torch, "amd_gpus", lambda: list(amd))
    monkeypatch.setattr(install_torch, "run", lambda cmd: "")
    monkeypatch.setattr(install_torch, "pip", lambda *a: calls.append(a) or 0)
    monkeypatch.setattr(sys, "version_info", py)
    return calls


def test_gtx_1070_gets_cuda_126(monkeypatch):
    calls = fake(monkeypatch, nvidia=[("NVIDIA GeForce GTX 1070", 8, "x")])
    assert install_torch.main() == 0
    assert calls == [("torch<2.15", "--index-url", "https://download.pytorch.org/whl/cu126")]


def test_rtx_50_gets_cuda_128(monkeypatch):
    calls = fake(monkeypatch, nvidia=[("NVIDIA GeForce RTX 5080", 16, "x")])
    install_torch.main()
    assert calls[0][-1].endswith("cu128")


def test_rx_9060_xt_gets_amd_rocm(monkeypatch):
    calls = fake(monkeypatch, amd=[("AMD Radeon(TM) Graphics", 0.5), ("AMD Radeon RX 9060 XT", 16)])
    assert install_torch.main() == 0
    assert len(calls) == 2
    assert all(u.startswith("https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/") for u in calls[0])
    assert any("torch-2.9.1" in u and "cp312" in u for u in calls[1])


def test_amd_needs_python_312(monkeypatch):
    calls = fake(monkeypatch, amd=[("AMD Radeon RX 9060 XT", 16)], py=(3, 13))
    assert install_torch.main() == 1 and calls == []


def test_old_amd_refused(monkeypatch):
    calls = fake(monkeypatch, amd=[("AMD Radeon RX 6700 XT", 12)])
    assert install_torch.main() == 1 and calls == []
