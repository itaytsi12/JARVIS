"""Files utilities: non-destructive filesystem helpers."""
from pathlib import Path
import os
import shutil
import ctypes
from uuid import UUID


def get_desktop_path() -> Path:
	"""Resolve the current user's real Desktop known folder, including redirection."""
	if os.name == "nt":
		folder_id = UUID("B4BFCC3A-DB2C-424C-B029-7FE99A87C641")
		guid = (ctypes.c_ubyte * 16).from_buffer_copy(folder_id.bytes_le)
		path_pointer = ctypes.c_wchar_p()
		result = ctypes.windll.shell32.SHGetKnownFolderPath(
			ctypes.byref(guid), 0, None, ctypes.byref(path_pointer)
		)
		if result == 0 and path_pointer.value:
			try:
				return Path(path_pointer.value)
			finally:
				ctypes.windll.ole32.CoTaskMemFree(path_pointer)
	return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


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


def create_text_file(path: str, contents: str, overwrite: bool = False) -> dict:
	p = Path(path).expanduser().resolve()
	if p.exists() and not overwrite:
		return {"success": False, "error": "File already exists; overwrite was not approved.", "path": str(p)}
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(contents, encoding="utf-8")
	return {"success": True, "message": f"Created {p}.", "path": str(p), "bytes": p.stat().st_size}


def read_text_file(path: str) -> dict:
	p = Path(path).expanduser().resolve()
	if not p.is_file():
		return {"success": False, "error": "File does not exist.", "path": str(p)}
	return {"success": True, "path": str(p), "contents": p.read_text(encoding="utf-8")}


def write_text_file(path: str, contents: str, overwrite: bool = False) -> dict:
	return create_text_file(path, contents, overwrite=overwrite)


def append_text_file(path: str, contents: str) -> dict:
	p = Path(path).expanduser().resolve()
	if not p.exists():
		return {"success": False, "error": "File does not exist.", "path": str(p)}
	with p.open("a", encoding="utf-8") as stream:
		stream.write(contents)
	return {"success": True, "message": f"Appended to {p}.", "path": str(p)}


def verify_file(path: str, expected_content: str | None = None) -> dict:
	p = Path(path).expanduser().resolve()
	if not p.is_file():
		return {"success": False, "message": "File verification failed.", "path": str(p), "error": "file_not_found"}
	if expected_content is not None:
		try:
			actual_content = p.read_text(encoding="utf-8")
		except (OSError, UnicodeDecodeError) as exc:
			return {"success": False, "message": "File content verification failed.", "path": str(p), "error": str(exc)}
		if actual_content != expected_content:
			return {"success": False, "message": "File content did not match.", "path": str(p), "error": "content_mismatch"}
	return {"success": True, "message": f"Verified {p}.", "path": str(p), "error": None}


def rename_path(path: str, new_name: str) -> dict:
	p = Path(path).expanduser().resolve()
	target = p.with_name(new_name)
	if target.exists():
		return {"success": False, "error": "Destination already exists."}
	p.rename(target)
	return {"success": True, "path": str(target), "message": f"Renamed to {target.name}."}


def copy_path(source: str, destination: str) -> dict:
	src, dst = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
	if dst.exists():
		return {"success": False, "error": "Destination already exists."}
	shutil.copy2(src, dst)
	return {"success": True, "path": str(dst), "message": f"Copied to {dst}."}


def move_path(source: str, destination: str) -> dict:
	src, dst = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
	if dst.exists():
		return {"success": False, "error": "Destination already exists."}
	shutil.move(str(src), str(dst))
	return {"success": True, "path": str(dst), "message": f"Moved to {dst}."}


def find_file(path: str, name: str) -> dict:
	root = Path(path).expanduser().resolve()
	matches = [str(item) for item in root.rglob(name)][:100]
	return {"success": True, "path": str(root), "items": matches}


def search_text(path: str, query: str) -> dict:
	root = Path(path).expanduser().resolve()
	matches = []
	for item in root.rglob("*"):
		if not item.is_file() or any(part.startswith(".venv") for part in item.parts):
			continue
		try:
			for line_number, line in enumerate(item.read_text(encoding="utf-8").splitlines(), 1):
				if query.lower() in line.lower():
					matches.append({"path": str(item), "line": line_number, "text": line[:200]})
					if len(matches) >= 100:
						return {"success": True, "matches": matches}
		except (UnicodeDecodeError, OSError):
			continue
	return {"success": True, "matches": matches}
