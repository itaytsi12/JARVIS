class TaskState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


class InvalidTransitionError(ValueError):
    pass


class Task:
    """A single task's own state machine. Deliberately correct -- the bug
    is not here, it's in how TaskManager coordinates multiple Tasks."""

    def __init__(self, task_id):
        self.task_id = task_id
        self.state = TaskState.PENDING

    def start(self):
        if self.state != TaskState.PENDING:
            raise InvalidTransitionError(f"cannot start from {self.state}")
        self.state = TaskState.RUNNING

    def pause(self):
        if self.state != TaskState.RUNNING:
            raise InvalidTransitionError(f"cannot pause from {self.state}")
        self.state = TaskState.PAUSED

    def resume(self):
        if self.state != TaskState.PAUSED:
            raise InvalidTransitionError(f"cannot resume from {self.state}")
        self.state = TaskState.RUNNING

    def complete(self):
        if self.state != TaskState.RUNNING:
            raise InvalidTransitionError(f"cannot complete from {self.state}")
        self.state = TaskState.COMPLETED
