from pathlib import Path
from datetime import datetime
from uuid import uuid4

from PIL import ImageGrab


SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


def take_screenshot() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    path = SCREENSHOT_DIR / f"screenshot_{timestamp}_{uuid4().hex[:8]}.png"

    image = ImageGrab.grab()
    image.save(path)

    verified=path.is_file() and path.stat().st_size>0
    return {"success":verified,"verified":verified,"path":str(path),"bytes":path.stat().st_size if verified else 0,"message":f"Saved screenshot to {path}." if verified else "Screenshot capture failed.","error":None if verified else "file_not_created"}
