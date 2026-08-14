import subprocess


CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def open_website(url: str) -> str:
    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        subprocess.Popen([CHROME_PATH, url])
        return f"Opened {url} in Google Chrome."

    except Exception as e:
        return f"Failed to open Chrome: {e}"