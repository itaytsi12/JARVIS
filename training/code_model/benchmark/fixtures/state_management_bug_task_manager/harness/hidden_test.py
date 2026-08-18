import pytest

from task_manager import TaskManager


def test_cannot_start_second_task_while_one_is_running_hidden():
    manager = TaskManager()
    manager.create_task("A")
    manager.create_task("B")
    manager.start_task("A")
    with pytest.raises(ValueError):
        manager.start_task("B")
    assert manager.running_task_ids() == ["A"]


def test_pause_then_start_another_task_works_hidden():
    """Sequence-dependent: pausing the active task must free up the
    scheduler to run a different one."""
    manager = TaskManager()
    manager.create_task("A")
    manager.create_task("B")
    manager.start_task("A")
    manager.pause_active_task()
    manager.start_task("B")
    assert manager.running_task_ids() == ["B"]


def test_resume_also_respects_single_active_task_hidden():
    manager = TaskManager()
    manager.create_task("A")
    manager.create_task("B")
    manager.start_task("A")
    manager.pause_active_task()
    manager.start_task("B")
    with pytest.raises(ValueError):
        manager.resume_task("A")
    assert manager.running_task_ids() == ["B"]


def test_full_realistic_sequence_hidden():
    manager = TaskManager()
    manager.create_task("A")
    manager.create_task("B")
    manager.create_task("C")
    manager.start_task("A")
    manager.pause_active_task()
    manager.start_task("B")
    manager.complete_active_task()
    manager.start_task("C")
    assert manager.running_task_ids() == ["C"]
    manager.pause_active_task()
    manager.resume_task("A")
    assert manager.running_task_ids() == ["A"]
