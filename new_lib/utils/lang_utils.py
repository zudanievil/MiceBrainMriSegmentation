import pathlib
from xml.etree import ElementTree


class UniversumSet:
    """instance always returns true for "in" operator"""
    def __contains__(self):
        return True


def values_to_path(d: dict):
    """
    For each key that endswith 'path' or 'folder', converts it's value
    to pathlib.Path. If value is None, skips it. Copies d.
    """
    p = pathlib.Path
    tk = ('_path', '_folder')
    return {k: p(v) if (k.endswith(tk) and v is not None) else v for k, v in d.items()}


def find_xml_node_parent(node: ElementTree.Element, tree_root: ElementTree.Element) -> 'ElementTree.Element or None':
    if node.attrib == tree_root.attrib:
        return None
    for potential_parent in tree_root.iter():
        if node in list(potential_parent):
            return potential_parent


def delete_folder(folder: pathlib.Path):
    for path in folder.iterdir():
        if path.is_dir():
            delete_folder(path)
        else:
            path.unlink()
    folder.rmdir()


def find_file(fname_start: str, folder: pathlib.Path) -> pathlib.Path:  # TODO: rewrite for fname_regex
    for p in folder.iterdir():
        if p.is_dir():
            result = find_file(fname_start, p)
            if result:
                return result
        if p.is_file() and p.name.startswith(fname_start):
            return p
