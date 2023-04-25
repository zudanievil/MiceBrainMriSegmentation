import os
import random
import pytest

from BrainS.lib.iterators import *
from BrainS.lib.trees import *


FIX_RANDOM_SEED = not os.getenv(
    "RANDOM_TESTS"
)  # disable randomness unless RANDOM_TESTS in environment
RANDOM_SEED = 42  # applies only when FIX_RANDOM_SEED
RANDOM_TEST_REPETITIONS = 100


def set_seed(seed=RANDOM_SEED) -> "rng state":
    state = random.getstate()
    if FIX_RANDOM_SEED:
        random.seed(seed)
    return state


def rand_loop(n_iter: int = RANDOM_TEST_REPETITIONS) -> Iterator[int]:
    for seed in range(n_iter, RANDOM_SEED + n_iter):
        set_seed(seed)
        yield seed


def mk_lst(*range_args):
    return list(range(*range_args))


def random_ftree(xs, n_braces=5):
    xs = xs.copy()
    left = Brace(None, left=True)  # type: ignore
    right = Brace(None, left=False)  # type: ignore
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
        lst = mk_lst(30)
        tree = random_tree(lst)
        ftree = list(flatten(tree))
        assert ftree == lst


def test_flatten_eq_unflatten():
    lst = mk_lst(30)
    for _ in rand_loop():
        tree = random_tree(lst)
        ftree = list(Flattener()(tree))
        tree_ = unflatten(ftree)[0]
        assert tree == tree_


def test_Tree_flatten_eq_unflatten():
    lst = mk_lst(30)
    for _ in rand_loop():
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
    t = random_Tree(mk_lst(30))
    rt = repr(t)
    ft = list(Tree.flatten(t))
    rft = repr_flat(ft)
    assert rft == rt
    print("\n=======\n", rt, "\n=======\n", sep="")


def test_Tree_filter():
    for _ in rand_loop():
        ft = random_ftree(mk_lst(30))
        t: Tree[int] = Tree.unflatten(ft)[0]
        ft2 = random_ftree(mk_lst(40, 50))
        ft3 = random_ftree(mk_lst(60, 70))
        pt2 = random.randint(1, len(ft) // 3 * 2)
        pt3 = random.randint(pt2, len(ft) - 1)
        while type(ft[pt2 - 1]) == Brace:
            pt2 += 1
        while type(ft[pt3 - 1]) == Brace:
            pt3 -= 1
        if pt2 > pt3:
            pt2, pt3 = pt3, pt2  # braces get messed up if pt2 > pt3
        ft_spliced = ft[:pt2] + ft2 + ft[pt2:pt3] + ft3 + ft[pt3:]
        pred = flat_tree_filter_lift(lambda x: x <= 30)
        ft_filt = list(filter(pred, ft_spliced))
        t_filt = Tree.unflatten(ft_filt)[0]
        assert t_filt == t


TAGS = "abcdefghiklmnopqrstuvwxyz0123456789"


def test_XML_flatten_eq_unflatten():
    tags = list(TAGS)
    for _ in rand_loop():
        tag_ftree = random_ftree(tags)
        tag_to_flat_node = flat_tree_lift(
            lambda t: XMLFlatNode(t, {"a": "x", "b": "y"})
        )
        xml_ftree = list(map(tag_to_flat_node, tag_ftree))
        xml = XMLTree.unflatten(xml_ftree)[0]
        xml_ftree2 = list(XMLTree.flatten(xml))
        xml2 = XMLTree.unflatten(xml_ftree2)[0]
        xml_ftree3 = list(XMLTree.flatten(xml2))
        assert xml_ftree2 == xml_ftree3
