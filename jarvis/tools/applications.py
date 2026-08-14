import subprocess
import time


def open_application(app_name: str) -> str:
    app_name = app_name.lower().strip()

    apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
    }

    if app_name in apps:
        try:
            print(f"[DEBUG] applications.open_application: launching {apps[app_name]}")
            t0 = time.perf_counter()
            proc = subprocess.Popen(apps[app_name])
            pid = None

            try:
                pid = proc.pid
            except Exception:
                pid = None

            print(f"[DEBUG] applications.open_application: Popen returned in {time.perf_counter() - t0:.3f}s, pid={pid}")

            # Return both a human message and the pid so callers can identify the launched process.
            return {
                "message": f"Opened {app_name} successfully.",
                "pid": pid,
            }
        except Exception as e:
            return f"Failed to open {app_name}: {e}"

    return f"I don't know how to open '{app_name}' yet."


def close_application(app_name: str) -> str:
    app_name = app_name.lower().strip()

    apps = {
        "notepad": "notepad.exe",
        "calculator": "CalculatorApp.exe",
        "paint": "mspaint.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "chrome": "chrome.exe",
        "spotify": "spotify.exe",
        "discord": "discord.exe",
    }

    process_name = apps.get(app_name)

    if not process_name:
        return f"I don't know how to close '{app_name}' yet."

    try:
        result = subprocess.run(
            ["taskkill", "/IM", process_name, "/F"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return f"Closed {app_name} successfully."

        return f"Could not close {app_name}: {result.stderr.strip()}"

    except Exception as e:
        return f"Failed to close {app_name}: {e}"