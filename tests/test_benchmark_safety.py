import sys
from unittest.mock import patch

from scripts import benchmark_web_answers


def test_web_benchmark_requires_explicit_run_flag(capsys):
    with patch.object(sys,"argv",["benchmark_web_answers.py"]),patch.object(benchmark_web_answers,"get_web_answer_service") as service:
        assert benchmark_web_answers.main()==2
    service.assert_not_called()
    assert "without --run" in capsys.readouterr().out
