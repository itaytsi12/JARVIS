"""Miscellaneous local utilities: time and date."""
from datetime import datetime


def get_time() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S")


def get_date() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d")


def get_day() -> str:
    now = datetime.now()
    return now.strftime("%A")
