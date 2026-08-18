from task_manager import TaskManager


def test_single_task_lifecycle():
    manager = TaskManager()
    manager.create_task("A")
    manager.start_task("A")
    assert manager.running_task_ids() == ["A"]
    manager.complete_active_task()
    assert manager.running_task_ids() == []
