import datetime


def log(message, level="INFO"):
    """Simple timestamped logger"""
    time = datetime.datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{time}] [{level}] {message}")