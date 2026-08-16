from pathlib import Path


def test_end_to_end_stt_requires_explicit_action_opt_in():
    source=(Path(__file__).resolve().parents[1]/"scripts"/"test_end_to_end_stt.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--execute-actions",action="store_true"' in source
    assert "if args.execute_actions:" in source

def test_audio_smoke_scripts_require_explicit_run_opt_in():
    root=Path(__file__).resolve().parents[1]
    for name in ("test_chatterbox_client.py","test_service_post.py"):
        source=(root/"scripts"/name).read_text(encoding="utf-8")
        assert 'add_argument("--run",action="store_true")' in source
        assert "without --run" in source
