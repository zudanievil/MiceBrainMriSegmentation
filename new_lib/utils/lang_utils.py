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
