import random

# import pytest
# from xml.etree.ElementTree import Element
# from typing import Iterable, TypeVar, NamedTuple
import pytest

from BrainS.lib.iterators import *
from BrainS.lib.iterators import _Brace


FIX_RANDOM_SEED = True  # disable for true randomness, enable for debug
RANDOM_SEED = 42  # applies only when FIX_RANDOM_SEED
RANDOM_TEST_REPETITIONS = 50


def set_seed(seed=RANDOM_SEED) -> "rng state":
    state = random.getstate()
    if FIX_RANDOM_SEED:
        random.seed(seed)
    return state


def rand_loop(n_iter: int = 100) -> Iterator[int]:
    for seed in range(n_iter, RANDOM_SEED + n_iter):
        set_seed(seed)
        yield seed


def random_ftree(xs, n_braces=5):
    xs = xs.copy()
    left = _Brace(None, left=True)  # type: ignore
    right = _Brace(None, left=False)  # type: ignore
    for i in range(n_braces):
        l = random.randint(0, len(xs) - 1)
        xs.insert(l, left)
        r = random.randint(l + 1, len(xs))
        xs.insert(r, right)
    xs.insert(0, left)
    xs.append(right)
    return xs


def random_tree(xs: list, n_braces=5):
    fxs = random_ftree(xs, n_braces)
    return unflatten(fxs)[0]


def random_Tree(xs: list, n_braces=5):
    fxs = random_ftree(xs, n_braces)
    return Tree.unflatten(fxs)[0]


def test_flatmap_eq_list():
    for _ in rand_loop():
        lst = list(range(30))
        tree = random_tree(lst)
        ftree = list(flatten(tree))
        assert ftree == lst


def test_flatten_eq_unflatten():
    for _ in rand_loop():
        lst = list(range(30))
        tree = random_tree(lst)
        ftree = list(Flattener()(tree))
        tree_ = unflatten(ftree)[0]
        assert tree == tree_


def test_Tree_flatten_eq_unflatten():
    for _ in rand_loop():
        lst = list(range(30))
        t = random_Tree(lst)
        ft2 = list(Tree.flatten(t))
        t2 = Tree.unflatten(ft2)[0]
        assert t == t2
        tree = Tree(1, [Leaf(2), Leaf(3), Tree(4, [Leaf(5), Leaf(6)])])
        ft = list(Tree.flatten(tree))
        tree2 = Tree.unflatten(ft)[0]
        ft2 = list(Tree.flatten(tree2))
        assert ft2 == ft
        assert tree2 == tree


@pytest.mark.visual
def test_Tree_repr():
    set_seed()
    lst = list(range(30))
    t = random_Tree(lst)
    rt = repr(t)
    ft = list(Tree.flatten(t))
    rft = Tree.flat_repr(ft)
    assert rft == rt
    print("\n=======\n", rt, "\n=======\n", sep="")


def test_Tree_filter():
    ...  # TODO
