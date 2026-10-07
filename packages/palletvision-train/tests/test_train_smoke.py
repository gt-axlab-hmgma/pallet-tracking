import pytest

from palletvision_train.cli import main


def test_cli_stub_exits_with_message() -> None:
    with pytest.raises(SystemExit):
        main()
