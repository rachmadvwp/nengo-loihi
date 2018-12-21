import pytest

from nengo_loihi.hardware import LoihiSimulator
from nengo_loihi.hardware import simulator as hardware_simulator


class MockNxsdk:
    def __init__(self):
        self.__version__ = None


def test_error_on_old_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.5.5"

    monkeypatch.setattr(hardware_simulator, 'nxsdk', mock)
    with pytest.raises(ImportError):
        LoihiSimulator.check_nxsdk_version()


def test_no_warn_on_current_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.7.0"

    monkeypatch.setattr(hardware_simulator, 'nxsdk', mock)
    with pytest.warns(None) as record:
        LoihiSimulator.check_nxsdk_version()
    assert len(record) == 0


def test_warn_on_future_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.7.1"

    monkeypatch.setattr(hardware_simulator, 'nxsdk', mock)
    with pytest.warns(UserWarning):
        LoihiSimulator.check_nxsdk_version()
