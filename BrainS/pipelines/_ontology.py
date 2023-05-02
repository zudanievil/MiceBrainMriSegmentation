from xml.etree.ElementTree import Element, ElementTree, parse as xml_parse

import PIL
import numpy as np

from ..lib.filesystem import FileTable
from ..prelude import *
from ..lib.iterators import flat_tree_lift, Brace
from ..lib.trees import Tree, XMLTree, XMLFlatNode

__all__ = [
    "Structure",
    "refactor_ontology",
    "Ontology",
    "read_ont",
    "write_ont",
    "MaskDir",
    # "ET_from_StructureTree",
]


class Structure(NamedTuple):
    name: str
    acronym: str
    id: str
    filename: str

    @classmethod
    def from_element(cls, elem: Element) -> "Structure":
        assert elem.tag == cls.NODE_TAG
        a = elem.attrib
        return cls(a["name"], a["acronym"], a["id"], a["filename"])

    NODE_TAG = "structure"

    def to_element(self) -> Element:
        return Element(self.NODE_TAG, self._asdict())

    def with_filename(self, fn: str) -> "Structure":
        return Structure(self.name, self.acronym, self.id, fn)


def refactor_ontology(old_root: Element) -> Element:
    """
    Makes xml ontology for Allen brain atlas more readable,
    filters out unimportant details (display options, etc).
    :param old_root
    :returns: new_root
    """

    new_root = __refactor_elem(old_root)
    stack = [
        (old_root, new_root),
    ]
    while stack:
        old, new = stack.pop()

        try:  # try to get node children
            old = old[10]
        except IndexError:  # means there are no children
            continue

        for sub_o in old:
            sub_n = __refactor_elem(sub_o)
            new.append(sub_n)
            stack.append((sub_o, sub_n))
    return new_root


def __refactor_elem(e: Element) -> Element:
    return Element(
        Structure.NODE_TAG,
        {
            "name": e[4]
            .text.lower()
            .strip('"')
            .replace(",", "")
            .replace("'", ""),
            "acronym": e[3].text.strip(' "'),
            "id": e[0].text,
            "filename": "",
        },
    )


_StructurePath = Tuple[Structure, ...]


class Ontology(NamedTuple):
    root: Tree[Structure]

    @classmethod
    def from_xml(cls, es: Element) -> "Ontology":
        cons = flat_tree_lift(Structure.from_element)
        tag = Structure.NODE_TAG
        tree = Tree.unflatten(cons(e) for e in XMLTree.flatten(es) if not isa(e, XMLFlatNode) or e.tag == tag)[0]
        return cls(tree)

    @staticmethod
    def from_xml_flat(es: Element) -> Iterator[Union[Structure, Brace]]:
        cons = flat_tree_lift(Structure.from_element)
        return (cons(e) for e in XMLTree.flatten(es))

    def to_xml(self) -> Element:
        decons = flat_tree_lift(Structure.to_element)
        return XMLTree.unflatten(decons(e) for e in Tree.flatten(self.root))[0]

    def iter(self) -> Iterator[Union[Structure, Brace]]:
        return Tree.flatten(self.root)

    def find(self, predicate) -> Opt[Structure]:
        for x in self.iter():
            if isa(x, Structure) and predicate(x):
                return x
        return None


def read_ont(path) -> Ontology:
    xml = xml_parse(path).getroot()
    return Ontology.from_xml(xml)


def write_ont(path, ont: Ontology):
    ElementTree(ont.to_xml()).write(path)


def read_mask(path) -> np.ndarray[bool]:
    mask = PIL.Image.open(path)
    mask = np.array(mask.getdata(), dtype=np.uint8) \
        .reshape((mask.size[1], mask.size[0]))
    return mask > 127


def MaskDir(dir) -> FileTable[Structure, np.ndarray[bool]]:
    dir = Path(dir)
    return FileTable(format=lambda s: dir / (s.filename + ".png"), _read=read_mask)

