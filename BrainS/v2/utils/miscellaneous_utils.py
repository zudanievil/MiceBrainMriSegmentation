import sys
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
    tk = ("_path", "_folder")
    return {
        k: p(v) if (k.endswith(tk) and v is not None) else v
        for k, v in d.items()
    }


def delete_folder(folder: pathlib.Path):
    for path in folder.iterdir():
        if path.is_dir():
            delete_folder(path)
        else:
            path.unlink()
    folder.rmdir()


def find_file(
    fname_start: str, folder: pathlib.Path
) -> pathlib.Path:  # TODO: rewrite for fname_regex
    for p in folder.iterdir():
        if p.is_dir():
            result = find_file(fname_start, p)
            if result:
                return result
        if p.is_file() and p.name.startswith(fname_start):
            return p


def list_substructures(
    root: "ElementTree.Element", names: "list[str]"
) -> "list[ElementTree.Element]":
    result = []
    for name in names:
        found = False
        for node in root.iter("structure"):
            if node.attrib["name"] == name:
                found = True
                break
        if found:
            for subnode in node.iter("structure"):
                result.append(subnode)
    return result


# def get_structure_parents(root: ElementTree.Element, structure_name: str) -> 'list[ElementTree.Element]':
#     if root.attrib['name'] == structure_name:
#         return [root]
#     for child in list(root):
#         result = get_structure_parents(child, structure_name)
#         if result:
#             result = [root] + result
#             return result


def get_structure_parents(
    root: ElementTree.Element, structure_name: str
) -> "list[ElementTree.Element]":
    result = []
    max_lvl = -1
    for node in root.iter("structure"):
        lvl = int(node.attrib["level"])
        if max_lvl < lvl:
            result.append(node)
            max_lvl += 1
        else:
            result[lvl] = node
        if node.attrib["name"] == structure_name:
            return result[: lvl + 1]
    raise KeyError(f"{structure_name} not in the tree")


def remove_node_xml_from_the_tree(
    root: ElementTree.Element, node: ElementTree.Element
):
    for p in root.iter():
        if node in list(p):
            p.remove(node)


def assert_path_is_short(p):
    if len(str(p)) > 100 and sys.platform.startswith("win"):
        raise AssertionError(
            f"{p} is longer than 100 chars, paths longer than 256 chars may "
            f"cause collisions on some systems (MS windows)"
        )
