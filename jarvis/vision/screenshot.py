from pathlib import Path
from datetime import datetime

from PIL import ImageGrab


SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


def take_screenshot() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"

    image = ImageGrab.grab()
    image.save(path)

    return str(path)