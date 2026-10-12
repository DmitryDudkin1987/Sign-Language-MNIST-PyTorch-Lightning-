import argparse


def str2bool(v):
    """Преобразует строку в bool для argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true"):
        return True
    if v.lower() in ("no", "false"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")