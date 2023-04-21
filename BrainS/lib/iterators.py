# from typing import Callable as _F
from typing import Generator
from itertools import *

from ..prelude import *


skipwhile = dropwhile


def as_iterator(
    i: Union[Iterable[T], Iterator[T]],
) -> Iterator[T]:
    return iter(i) if not hasattr(i, "__next__") else i


def is_iterator(any) -> bool:
    return hasattr(any, "__next__")


def is_iterable(any) -> bool:
    return hasattr(any, "__iter__")


def foreach(itr: Iterable) -> None:
    for _ in itr:
        pass


def take(n: int, itr: Iterable[T]) -> Iterator[T]:
    return islice(itr, 0, n)


def skip(n: int, itr: Iterable[T]) -> Iterator[T]:
    return islice(itr, n, None)


def _identity(x):
    return x


def filtermap(
    fn: Fn[[T], Opt[T1]],
    xs: Iterable[T],
) -> Iterator[T1]:
    for x in xs:
        y = fn(x)
        if y is None:
            continue
        yield y


def zip_w_next(itr: Iterable[T]) -> Iterator[Tuple[T, T]]:
    first = True
    prev = None
    for i in itr:
        if first:
            first = False
            prev = i
            continue
        yield (prev, i)
        prev = i


def infinity(start=0, step=1):
    while True:
        yield start
        start = start + step


def unfold(
    start: T,
    update: Fn[[T], T],
    *,
    yield_first=False,
    sentinel: T = None,
) -> Iterator[T]:
    """
    apply `start = update(start); yield start` until `start==sentinel`.
    try it out:
    ```
    fib10 = list(itr.unfold(
        (1, 1), lambda x: (x[1], x[0] + x[1]), sentinel = (144, 233)))
    ```
    """
    if yield_first:
        yield start
    while True:
        start = update(start)
        if start == sentinel:
            break
        yield start


# may be fun to implement:
# https://towardsdatascience.com/python-stack-frames-and-tail-call-optimization-4d0ea55b0542


class _Brace(NamedTuple):
    """for flattened recursive objects"""

    level: int
    left: bool = True

    def __repr__(self) -> str:
        return f"{self.level}{'{' if self.left else '}'}"


_FlatTree = Union[T, _Brace]
_LeafT = TypeVar("_LeafT")
_NodeT = TypeVar("_NodeT")
_Tree = Union[_LeafT, _NodeT]


class Flattener(NamedTuple):
    """
    with braces == False, flatten tree into plain iterator
    try it:
    ```
    xs = [1, 2, [3, 4, 5], 6, [7, 8, [9, 10, [11]], 12], 13]
    for i in flatten(xs):
        print(i)
    ```
    uses loops instead of recursion:
    ```
    xs = [5, ]
    sys.setrecursionlimit(100)
    for _ in range(200):
        xs = [xs, ]
    for i in flatten(xs):
        print(i)
    ```
    if braces == True, tree is serialized,
    _Brace objects are added to separate subtrees.
    To better understand how it works,
    imagine how you would serialize a lisp AST into token stream.
    ----
    these parameters are needed when you use tree objects
    (like xml.etree) instead of lists / tuples:
    :param has_children: should return False if x is a leaf
    (no children), True otherwise (has children)
    :param get_children: when nodes have values inside them
    (like attributes in xml trees) should yield
    node inner value (tag+attributes in xml case),
    then child nodes. This may be counter-intuitive at first,
    but remember that you are "serializing lisp AST".
    :param leaf_unwrap: unwrap leaf containers with
    this one (returns tag+attributes in xml case)
    """

    braces: bool = True
    has_children: Fn[[_Tree], bool] = is_iterable
    get_children: Fn[[_NodeT], Iterator[_Tree]] = as_iterator
    leaf_unwrap: Fn[[_LeafT], T] = _identity

    def __call__(self, xs: _Tree) -> Iterator[_FlatTree]:
        get_children = self.get_children
        leaf_unwrap = self.leaf_unwrap
        has_children = self.has_children
        braces = self.braces

        if not has_children(xs):
            if braces:
                yield from [_Brace(0, True), leaf_unwrap(xs), _Brace(0, False)]
                return
        stack = [
            get_children(xs),
        ]
        if braces:
            yield _Brace(0, True)
        while stack:
            xs = stack[-1]
            lvl = len(stack)
            for x in xs:
                if has_children(x):
                    if braces:
                        yield _Brace(lvl, True)
                    stack.append(get_children(x))
                    break
                else:
                    yield leaf_unwrap(x)
            else:  # loop finished with no exceptions or breaks
                stack.pop()
                if braces:
                    yield _Brace(lvl - 1, False)


flatten = Flattener(braces=False)

# def flat_tree_filter(
#     predicate: Fn[[T], bool], xs: Iterable[_FlatTree]
# ) -> Iterator[_FlatTree]:
#     """
#     if first element after brace is 'bad', then the whole
#     subtree is removed, if second-to-last, then only this element is removed.
#     tree is treated as lisp AST, first element after
#     left brace is a subtree root value, other values are children
#     """
#     bad_braces = 0  # on one hand, this function is very
#     # procedural, on the other there is only one state variable (not counting the iterator)
#     xs = as_iterator(xs)
#     for x in xs:
#         if type(x) != _Brace:
#             if bad_braces == 0 and predicate(x):
#                 yield x
#             continue
#         # x is a _Brace
#         if x.left:
#             if bad_braces != 0:
#                 bad_braces += 1
#                 continue
#             head = next(xs)  # lookahead needed
#             if predicate(head):
#                 yield x
#                 yield head
#             else:
#                 bad_braces = 1
#         else:  # right
#             if bad_braces == 0:
#                 yield x
#             else:
#                 bad_braces -= 1


def flat_tree_lift(f: Fn[[T], T1]) -> Fn[[_FlatTree], Union[T1, _Brace]]:
    """lift f to operate on flat tree type union.
    can be used with ordinary `map`"""

    def clj(x: _FlatTree) -> Union[T1, _Brace]:
        return x if type(x) == _Brace else f(x)

    return clj


class flat_tree_filter_lift:  # TODO: test
    """
    lift ``predicate`` to ``filter`` flat tree.

    if first element after brace is 'bad', then the whole
    subtree is removed, if second-to-last, then only this element is removed.
    tree is treated as lisp AST, first element after
    left brace is a subtree root value, other values are children.
    leaves a brace pair in place of a removed subtree
    """

    __slots__ = "predicate", "bad_braces", "next_is_head"

    def __init__(self, predicate: Fn[[T], bool]):
        self.predicate = predicate
        self.bad_braces = 0
        self.next_is_head = False

    __repr__ = repr_slots

    def __call__(self, x: _FlatTree) -> bool:
        if type(x) == _Brace:
            if self.bad_braces != 0:
                self.bad_braces += +1 if x.left else -1
                return (
                    self.bad_braces == 0
                )  # True if that was last 'bad' right brace
            else:
                self.next_is_head = x.left
                return True
        else:  # x is a node
            if self.bad_braces != 0:
                return False
            pred = self.predicate(x)
            if self.next_is_head:
                if pred:
                    self.bad_braces += 1
                self.next_is_head = False
            return pred


class Unflattener(NamedTuple):
    """
    unflatten the tree that was flattened with
    ``flatten(tree, braces=True)`` does not check brace levels.

    ``
    xs = [1, 2, [3, 4, 5], 6, [7, 8, [9, 10, [11, ]]], 12]
    fxs = [x for x in flatten(xs, braces=True)]
    assert xs == unflatten(fxs)[0] # several trees can be deserialized at once
    ``

    if you want to recieve trees of custom type `T1`,
    you should supply constructors `T -> T1` and `List[T | T1] -> T1`
    (if you dont get it, try `str` and `tuple` respectively with the example above).
    """

    tree_cons: Fn[[List[Union[T, T1]]], T1] = list
    leaf_cons: Fn[[T], T1] = _identity

    def lazy_collector(self) -> Generator[List[_Tree], Opt[_FlatTree], None]:
        """
        gradually collect tree via
        ```
        c = Unflattener(...).lazy_collector()
        c.send(None)
        for x in xs:
            c.send(x)
        tree c.send(Unflattener.COLLECTOR_SENTINEL)
        ```
        """
        # send generator. viewer discretion is advised
        tree_cons = self.tree_cons
        leaf_cons = self.leaf_cons
        root_hook = []
        stack = [
            root_hook,
        ]
        while True:
            x = yield
            if x is self.COLLECTOR_SENTINEL:
                break
            if type(x) == _Brace:
                if x.left:
                    stack.append([])
                else:
                    tree = tree_cons(stack.pop())
                    stack[-1].append(tree)
            else:
                stack[-1].append(leaf_cons(x))
        assert len(stack) == 1, "number of braces does not match"
        yield root_hook

    COLLECTOR_SENTINEL = object()

    def __call__(self, xs: Iterable[_FlatTree]) -> List[_Tree]:
        """shorthand for collecting iterable into trees"""
        c = self.lazy_collector()
        c.send(None)
        for x in xs:
            c.send(x)
        return c.send(self.COLLECTOR_SENTINEL)


unflatten = Unflattener()


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
        if not xs:
            return xs  # type: ignore  # skip singleton list
        return Tree(
            xs[0],
            [
                x if type(x) == Tree else Leaf(x)
                for x in skip(1, xs)
                if x  # skip emty lists
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

    @staticmethod
    def flat_repr(xs: Iterable[_FlatTree]) -> str:
        lvl = 0
        next_is_head = False
        lines = []
        for x in xs:
            if type(x) == _Brace:
                lvl += +1 if x.left else -1
                next_is_head = x.left
            else:
                if next_is_head:
                    pad = "  " * (lvl - 1) + "->"
                    next_is_head = False
                else:
                    pad = "  " * lvl
                lines.append(pad + repr(x))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return Tree.flat_repr(Tree.flatten(self))


def Leaf(data: T) -> Tree[T]:
    return Tree(data)
