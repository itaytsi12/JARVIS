from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.autostart import TASK_NAME, background_python, install_autostart


if __name__ == "__main__":
    install_autostart()
    print(f"Installed {TASK_NAME}.")
    print(f"Background Python: {background_python()}")
