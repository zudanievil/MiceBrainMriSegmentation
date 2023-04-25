from ..prelude import *
from BrainS.lib.iterators import *
from xml.etree.ElementTree import Element

__all__ = [
    "Tree",
    "Leaf",
    "XMLFlatNode",
    "XMLTree",
]


class Tree(Generic[T]):
    """singly-linked tree container that can hold arbitrary data types"""

    __slots__ = ("data", "children")

    def __init__(self, data: T, children: List["Tree[T]"] = None):
        self.data = data
        self.children = [] if children is None else children

    def unwrap(self: Union[T, "Tree[T]"]) -> T:
        return self.data if type(self) is Tree else self

    def head_tail_iter(self) -> Iterator[T]:
        yield self.data
        for c in self.children:
            yield c

    def has_children(self: Union[T, "Tree[T]"]) -> bool:
        return type(self) is Tree and bool(self.children)

    @staticmethod
    def head_tail_cons(xs: List[Union[T, "Tree[T]"]]) -> "Tree[T]":
        return Tree(
            xs[0],
            [
                x if type(x) == Tree else Leaf(x)
                for x in skip(1, xs)
                # if x  # skip empty lists
            ],
        )

    # for when flattened methods break
    # __repr__ = repr_slots
    # def __eq__(self, other) -> bool:
    #     return self.data == other.data and self.children == other.children

    flatten = Flattener(
        has_children=has_children,
        get_children=head_tail_iter,
        leaf_unwrap=unwrap,
    )
    unflatten = Unflattener(head_tail_cons.__func__)  # type: ignore  #  static methods are containers

    def __eq__(self, other) -> bool:
        flatt = Tree.flatten
        return any(x == y for x, y in zip(flatt(self), flatt(self)))

    def __repr__(self) -> str:
        return repr_flat(Tree.flatten(self))


def Leaf(data: T) -> Tree[T]:
    return Tree(data)


class XMLFlatNode(NamedTuple):
    tag: str
    attrib: Dict[str, str]

    @classmethod
    def from_xml(cls, x: Element) -> "XMLFlatNode":
        return cls(x.tag, x.attrib)

    def to_xml(self) -> Element:
        assert type(self) == XMLFlatNode  # for debug
        return Element(self.tag, self.attrib)

    def __repr__(self):
        d = " ".join(f'{k}="{v}"' for k, v in self.attrib.items())
        return f"<{self.tag} {d}/>"


class XMLTree:
    @staticmethod
    def has_children(x: Element):
        return type(x) == Element and len(x) != 0

    @staticmethod
    def head_tail_iter(xs: Element) -> Iterator[XMLFlatNode]:
        yield XMLFlatNode.from_xml(xs)
        yield from xs

    flatten = Flattener(
        has_children=has_children.__func__,  # type: ignore
        get_children=head_tail_iter.__func__,  # type: ignore
        leaf_unwrap=XMLFlatNode.from_xml,  # type: ignore
    )

    @staticmethod
    def head_tail_cons(xs: List[Element]) -> Element:
        e = xs[0]
        e.extend(skip(1, xs))
        return e

    unflatten = Unflattener(
        tree_cons=head_tail_cons.__func__,  # type: ignore
        leaf_cons=XMLFlatNode.to_xml,
    )
