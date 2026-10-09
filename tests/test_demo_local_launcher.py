import socket

import pytest

from scripts.run_upgrade_demo_local import arguments, launch_command, require_free_port, stop_owned


@pytest.mark.parametrize("argv", [["--port", "0"], ["--port", "65536"], ["--minutes", "0"], ["--minutes", "61"], ["--minutes", "nan"]])
def test_invalid_port_or_unbounded_duration_rejected(argv):
    with pytest.raises(SystemExit):
        arguments(argv)


def test_command_forces_local_synthetic_entrypoint_and_no_telemetry():
    args = arguments([])
    assert args.minutes == 45 and args.port == 8502
    command = launch_command(args.port)
    assert "app_upgrade.py" in command and "app.py" not in command
    assert "--server.address=127.0.0.1" in command and "--browser.gatherUsageStats=false" in command


def test_busy_port_fails_without_touching_existing_listener():
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0)); owner.listen()
        with pytest.raises(RuntimeError, match="chọn --port khác"):
            require_free_port(owner.getsockname()[1])
        assert owner.fileno() >= 0


def test_already_stopped_child_is_not_terminated():
    class Child:
        def poll(self): return 0
        def terminate(self): raise AssertionError("No termination after exit")
    stop_owned(Child())
