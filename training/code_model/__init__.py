"""Real LoRA/QLoRA training backend, benchmark harness, and CLI for
JARVIS's local coding student model.

This package is downstream of, and reuses, the continual-learning pipeline
in `brain/learning_*.py` and `voice/learning_approval.py` -- it does not
duplicate approval, dataset-versioning, or model-registry concerns, only
implements the previously-fake `TrainingBackend`/`Benchmark` protocols for
real. See CLAUDE.md's "Voice-approved continual learning" section.
"""
