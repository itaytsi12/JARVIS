"""Files utilities: non-destructive filesystem helpers."""
from pathlib import Path
import os
import shutil
import ctypes
import time
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


def get_documents_path() -> Path:
	if os.name == "nt":
		folder_id = UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7")
		guid = (ctypes.c_ubyte * 16).from_buffer_copy(folder_id.bytes_le)
		path_pointer = ctypes.c_wchar_p()
		result = ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(path_pointer))
		if result == 0 and path_pointer.value:
			try:return Path(path_pointer.value)
			finally:ctypes.windll.ole32.CoTaskMemFree(path_pointer)
	return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents"


def open_path(path: str) -> dict:
	p = Path(path).expanduser().resolve()
	if not p.exists():return {"success":False,"message":"The requested path does not exist.","path":str(p),"error":"path_not_found"}
	try:
		os.startfile(str(p))
		if p.suffix.casefold()==".txt":
			from tools.window import find_application_windows
			deadline=time.perf_counter()+6;user32=ctypes.windll.user32
			while time.perf_counter()<deadline:
				for hwnd in find_application_windows("notepad"):
					length=user32.GetWindowTextLengthW(hwnd);title=ctypes.create_unicode_buffer(length+1);user32.GetWindowTextW(hwnd,title,length+1)
					if p.name.casefold() in title.value.casefold():return {"success":True,"verified":True,"message":f"Opened {p.name}.","path":str(p),"hwnd":hwnd}
				time.sleep(.05)
			return {"success":False,"verified":False,"message":f"Started opening {p.name}, but no matching Notepad window appeared.","path":str(p),"error":"path_window_unverified"}
		return {"success":True,"verified":False,"message":f"Opened {p.name}; the associated application was not independently verified.","path":str(p)}
	except Exception as exc:return {"success":False,"message":f"Failed to open {p.name}.","path":str(p),"error":str(exc)}


def open_known_folder(name: str) -> dict:
	name = name.lower().strip()

	user = os.path.expanduser("~")

	mapping = {
		"downloads": Path(user) / "Downloads",
		"documents": get_documents_path(),
		"desktop": get_desktop_path(),
		"pictures": Path(user) / "Pictures",
		"music": Path(user) / "Music",
		"videos": Path(user) / "Videos",
	}

	path = mapping.get(name)

	if not path:
		return {"success":False,"message":f"Unknown known folder: {name}","error":"unknown_folder"}

	try:
		os.startfile(str(path))
		return {"success":True,"message":f"Opened {name} folder.","path":str(path)}
	except Exception as e:
		return {"success":False,"message":f"Failed to open {name}.","error":str(e)}


def list_files(path: str) -> dict:
	p = Path(path).expanduser()

	if not p.exists():
		return {"success": False, "error": "Path does not exist."}

	if not p.is_dir():
		return {"success": False, "error": "Path is not a directory."}

	items = [str(x.name) for x in p.iterdir()]

	return {"success": True,"verified":True, "path": str(p), "items": items}


def exists(path: str) -> dict:
	p = Path(path).expanduser()
	return {"success":True,"verified":True,"exists": p.exists(), "path": str(p)}


def create_text_file(path: str, contents: str, overwrite: bool = False) -> dict:
	p = Path(path).expanduser().resolve()
	if p.exists() and not overwrite:
		return {"success": False, "error": "File already exists; overwrite was not approved.", "path": str(p)}
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(contents, encoding="utf-8")
	return {"success": True, "verified": True, "message": f"Created {p}.", "path": str(p), "bytes": p.stat().st_size}


def read_text_file(path: str) -> dict:
	p = Path(path).expanduser().resolve()
	if not p.is_file():
		return {"success": False, "error": "File does not exist.", "path": str(p)}
	contents = p.read_text(encoding="utf-8")
	return {"success": True, "verified": True, "path": str(p), "contents": contents, "message": contents}


def write_text_file(path: str, contents: str, overwrite: bool = False) -> dict:
	return create_text_file(path, contents, overwrite=overwrite)


def append_text_file(path: str, contents: str) -> dict:
	p = Path(path).expanduser().resolve()
	if not p.exists():
		return {"success": False, "error": "File does not exist.", "path": str(p)}
	with p.open("a", encoding="utf-8") as stream:
		stream.write(contents)
	verified=p.read_text(encoding="utf-8").endswith(contents)
	return {"success": verified,"verified":verified,"message": f"Appended to {p}." if verified else "Append verification failed.", "path": str(p),"error":None if verified else "content_mismatch"}


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
	return {"success": True, "verified": True, "message": f"Verified {p}.", "path": str(p), "error": None}


def rename_path(path: str, new_name: str) -> dict:
	p = Path(path).expanduser().resolve()
	target = p.with_name(new_name)
	if target.exists():
		return {"success": False, "error": "Destination already exists."}
	p.rename(target)
	verified=target.exists() and not p.exists()
	return {"success":verified,"verified":verified,"path":str(target),"message":f"Renamed to {target.name}." if verified else "Rename verification failed.","error":None if verified else "verification_failed"}


def copy_path(source: str, destination: str) -> dict:
	src, dst = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
	if dst.exists():
		return {"success": False, "error": "Destination already exists."}
	shutil.copy2(src, dst)
	verified=dst.exists() and src.exists()
	return {"success":verified,"verified":verified,"path":str(dst),"message":f"Copied to {dst}." if verified else "Copy verification failed.","error":None if verified else "verification_failed"}


def move_path(source: str, destination: str) -> dict:
	src, dst = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
	if dst.exists():
		return {"success": False, "error": "Destination already exists."}
	shutil.move(str(src), str(dst))
	verified=dst.exists() and not src.exists()
	return {"success":verified,"verified":verified,"path":str(dst),"message":f"Moved to {dst}." if verified else "Move verification failed.","error":None if verified else "verification_failed"}


def find_file(path: str, name: str) -> dict:
	root = Path(path).expanduser().resolve()
	matches = [str(item) for item in root.rglob(name)][:100]
	return {"success": True,"verified":True, "path": str(root), "items": matches}


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
						return {"success": True,"verified":True, "matches": matches}
		except (UnicodeDecodeError, OSError):
			continue
	return {"success": True,"verified":True, "matches": matches}


def create_directory(path: str) -> dict:
	"""Create a directory (and any missing parents).

	Succeeds quietly when the directory already exists -- that is the
	desired end state, not an error -- but says which case happened so a
	caller is never misled about whether it created anything.
	"""
	p = Path(path).expanduser().resolve()
	if p.is_file():
		return {"success": False, "message": "A file already exists at that path.", "error": "path_is_file", "path": str(p)}
	already = p.is_dir()
	try:
		p.mkdir(parents=True, exist_ok=True)
	except OSError as exc:
		return {"success": False, "message": "The directory could not be created.", "error": str(exc), "path": str(p)}
	created = p.is_dir()
	return {
		"success": created,
		"verified": created,
		"created": created and not already,
		"already_existed": already,
		"message": (f"{p} already exists." if already else f"Created {p}.") if created else "Directory creation could not be verified.",
		"path": str(p),
		"error": None if created else "verification_failed",
	}

# ---------------------------------------------------------------------------
# Discovery: metadata, and "the file I was working on".
# ---------------------------------------------------------------------------
#: Where `recent_files` looks when no path is given. These are the places a
#: person actually saves work; the whole user profile is deliberately NOT
#: scanned, because AppData alone would swamp the result with cache churn
#: that no one ever means by "the file I was working on".
DEFAULT_RECENT_ROOTS = ("desktop", "documents", "downloads")

#: Never surfaced as "recent work": these change constantly for reasons
#: that have nothing to do with the user. Shared with tools/code.py's
#: pruning vocabulary rather than re-invented.
_RECENT_SKIP_SUFFIXES = {".tmp", ".log", ".lock", ".pyc", ".pyo", ".bak", ".crdownload", ".part"}

#: Hard bound on the walk. A deep tree must not turn one question into a
#: minutes-long traversal -- `recent_files` is meant to answer in about a
#: second, and it reports honestly when it hit the bound.
MAX_RECENT_SCAN = 40_000


def _known_root(name: str) -> Path | None:
	name = (name or "").strip().lower()
	if name == "desktop":
		return get_desktop_path()
	if name in {"documents", "docs"}:
		return get_documents_path()
	if name in {"downloads", "download"}:
		return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Downloads"
	if name in {"pictures", "photos"}:
		return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Pictures"
	if name in {"home", "profile", "user"}:
		return Path(os.environ.get("USERPROFILE", str(Path.home())))
	return None


def file_info(path: str) -> dict:
	"""Size, timestamps and kind for one path.

	Answers "when did I last touch this", "how big is it" and "is this a
	folder" in one call, so the agent does not have to shell out to
	PowerShell for something the standard library already knows.
	"""
	p = Path(path).expanduser()
	try:
		p = p.resolve()
		stat = p.stat()
	except OSError as exc:
		return {"success": False, "message": f"I could not read {path}.", "error": f"{type(exc).__name__}", "path": str(p)}

	is_dir = p.is_dir()
	entries = None
	if is_dir:
		try:
			entries = sum(1 for _ in p.iterdir())
		except OSError:
			entries = None
	modified = time.localtime(stat.st_mtime)
	return {
		"success": True,
		"verified": True,
		"path": str(p),
		"name": p.name,
		"kind": "directory" if is_dir else "file",
		"suffix": "" if is_dir else p.suffix.lower(),
		"size_bytes": None if is_dir else stat.st_size,
		"size_kb": None if is_dir else round(stat.st_size / 1024, 1),
		"entries": entries,
		"modified": time.strftime("%Y-%m-%d %H:%M:%S", modified),
		"modified_epoch": stat.st_mtime,
		"age_hours": round((time.time() - stat.st_mtime) / 3600, 1),
		"created_epoch": stat.st_ctime,
		"read_only": not os.access(p, os.W_OK),
		"message": (
			f"{p.name} is a folder with {entries} item{'' if entries == 1 else 's'}, last changed {time.strftime('%Y-%m-%d %H:%M', modified)}."
			if is_dir
			else f"{p.name} is {round(stat.st_size / 1024, 1)} KB, last modified {time.strftime('%Y-%m-%d %H:%M', modified)}."
		),
	}


def recent_files(
	path: str | None = None,
	within_hours: float = 48.0,
	limit: int = 25,
	suffixes: list[str] | str | None = None,
) -> dict:
	"""Files changed recently, newest first.

	This is what answers "find the file I worked on yesterday": the agent
	gets real, ranked candidates with timestamps instead of having to
	guess a filename. `path` may be a real directory OR a known-folder
	name ("desktop", "documents", "downloads"); with no path, all three
	are scanned.

	Directories that are noise by construction (`.git`, `node_modules`,
	virtualenvs, caches -- `tools/code.py`'s existing vocabulary) are
	pruned during the walk rather than filtered afterwards, for the same
	reason `walk_source_files` exists: `rglob` cannot prune and descending
	into a virtualenv to throw the results away costs orders of magnitude
	more than the answer is worth.
	"""
	from tools.code import _ignored_directory  # one pruning vocabulary, not two

	roots: list[Path] = []
	if path:
		known = _known_root(path)
		roots = [known] if known is not None else [Path(path).expanduser()]
	else:
		roots = [root for root in (_known_root(name) for name in DEFAULT_RECENT_ROOTS) if root is not None]

	wanted_suffixes: set[str] | None = None
	if suffixes:
		if isinstance(suffixes, str):
			suffixes = [suffixes]
		wanted_suffixes = {("." + item.lstrip(".")).lower() for item in suffixes if item}

	cutoff = time.time() - max(0.0, float(within_hours)) * 3600
	found: list[dict] = []
	scanned = 0
	truncated = False
	missing_roots: list[str] = []

	for root in roots:
		try:
			if not root.is_dir():
				missing_roots.append(str(root))
				continue
		except OSError:
			missing_roots.append(str(root))
			continue
		for directory, subdirectories, filenames in os.walk(root):
			subdirectories[:] = [name for name in subdirectories if not _ignored_directory(name) and not name.startswith(".")]
			for filename in filenames:
				scanned += 1
				if scanned > MAX_RECENT_SCAN:
					truncated = True
					break
				suffix = Path(filename).suffix.lower()
				if suffix in _RECENT_SKIP_SUFFIXES or filename.startswith("~$"):
					continue
				if wanted_suffixes is not None and suffix not in wanted_suffixes:
					continue
				full = Path(directory) / filename
				try:
					modified = full.stat().st_mtime
				except OSError:
					continue
				if modified < cutoff:
					continue
				found.append({
					"path": str(full),
					"name": filename,
					"modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(modified)),
					"modified_epoch": modified,
					"age_hours": round((time.time() - modified) / 3600, 1),
				})
			if truncated:
				break
		if truncated:
			break

	found.sort(key=lambda item: item["modified_epoch"], reverse=True)
	limit = max(1, min(int(limit or 25), 200))
	shown = found[:limit]
	if shown:
		message = f"{len(found)} file{'' if len(found) == 1 else 's'} changed in the last {int(within_hours)} hours; newest is {shown[0]['name']}."
	else:
		message = f"Nothing was changed in the last {int(within_hours)} hours under {', '.join(str(root) for root in roots) or 'those folders'}."
	return {
		"success": True,
		"verified": True,
		"message": message,
		"items": shown,
		"matched": len(found),
		"scanned": scanned,
		"roots": [str(root) for root in roots],
		"missing_roots": missing_roots,
		"scan_truncated": truncated,
	}
