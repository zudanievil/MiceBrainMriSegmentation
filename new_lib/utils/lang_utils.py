import pathlib


class UniversumSet:
    """instance always returns true for "in" operator"""
    def __contains__():
        return True


def values_to_path(d: dict):
    """
    For each key that endswith 'path' or 'folder', converts it's value
    to pathlib.Path. If value is None, skips it. Copies d.
    """
    p = pathlib.Path
    tk = ('_path', '_folder')
    return {k: p(v) if (k.endswith(tk) and v!=None) else v for k, v in d.items()}
