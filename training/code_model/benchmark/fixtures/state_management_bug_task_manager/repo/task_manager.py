from task_state import Task, TaskState


class TaskManager:
    """Coordinates many Task objects. Intended invariant: at most one task
    may be RUNNING at any given time (a single-worker scheduler)."""

    def __init__(self):
        self.tasks = {}
        self.active_task_id = None

    def create_task(self, task_id):
        task = Task(task_id)
        self.tasks[task_id] = task
        return task

    def running_task_ids(self):
        return [tid for tid, task in self.tasks.items() if task.state == TaskState.RUNNING]

    def start_task(self, task_id):
        # BUG: doesn't check whether another task is already RUNNING before
        # starting this one.
        self.tasks[task_id].start()
        self.active_task_id = task_id

    def pause_active_task(self):
        if self.active_task_id is None:
            return
        self.tasks[self.active_task_id].pause()
        self.active_task_id = None

    def resume_task(self, task_id):
        # BUG: same missing check as start_task.
        self.tasks[task_id].resume()
        self.active_task_id = task_id

    def complete_active_task(self):
        if self.active_task_id is None:
            raise ValueError("no active task to complete")
        self.tasks[self.active_task_id].complete()
        self.active_task_id = None
