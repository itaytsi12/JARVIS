"""Files utilities: non-destructive filesystem helpers."""
from pathlib import Path
import os


def open_known_folder(name: str) -> str:
	name = name.lower().strip()

	user = os.path.expanduser("~")

	mapping = {
		"downloads": Path(user) / "Downloads",
		"documents": Path(user) / "Documents",
		"desktop": Path(user) / "Desktop",
		"pictures": Path(user) / "Pictures",
		"music": Path(user) / "Music",
		"videos": Path(user) / "Videos",
	}

	path = mapping.get(name)

	if not path:
		return f"Unknown known folder: {name}"

	try:
		os.startfile(str(path))
		return f"Opened {name} folder."
	except Exception as e:
		return f"Failed to open {name}: {e}"


def list_files(path: str) -> dict:
	p = Path(path).expanduser()

	if not p.exists():
		return {"success": False, "error": "Path does not exist."}

	if not p.is_dir():
		return {"success": False, "error": "Path is not a directory."}

	items = [str(x.name) for x in p.iterdir()]

	return {"success": True, "path": str(p), "items": items}


def exists(path: str) -> dict:
	p = Path(path).expanduser()
	return {"exists": p.exists(), "path": str(p)}
