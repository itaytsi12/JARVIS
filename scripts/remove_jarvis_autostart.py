from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.autostart import TASK_NAME, remove_autostart


if __name__ == "__main__":
    remove_autostart()
    print(f"Removed {TASK_NAME}.")
