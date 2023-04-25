from xml.etree.ElementTree import Element

from ..prelude import *
from ..lib.iterators import flat_tree_lift
from ..lib.trees import Tree, XMLTree

__all__ = [
    "Structure",
    "refactor_ontology",
    "ET_to_StructureTree",
    "ET_from_StructureTree",
]


class Structure(NamedTuple):
    name: str
    acronym: str
    id: str
    filename: str

    @classmethod
    def from_element(cls, elem: Element) -> "Structure":
        return cls(**elem.attrib)

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


def ET_to_StructureTree(es: Element) -> Tree[Structure]:
    cons = flat_tree_lift(Structure.from_element)
    return Tree.unflatten(cons(e) for e in XMLTree.flatten(es))[0]


def ET_from_StructureTree(st: Tree[Structure]) -> Element:
    decons = flat_tree_lift(Structure.to_element)
    return XMLTree.unflatten(decons(e) for e in Tree.flatten(st))[0]
