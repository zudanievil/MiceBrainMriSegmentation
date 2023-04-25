import pytest

from BrainS.prelude import *
from BrainS.lib.iterators import Flattener, foreach
from BrainS.lib.filesystem import (
    iter_tree_braces,
    File,
    FileTable,
    PrefixSuffixFormatter,
)
import BrainS.lib.io_utils as io
import numpy as np


@pytest.mark.visual
def test_file_iter_tree(tmp_path_factory):
    """
    this test is badly designed. moreover, it can fail
    due to ``Path.iterdir()`` result variation.
    use common sense to interpret the results.
    """
    rt = tmp_path_factory.mktemp("iter_tree")
    a = rt / "a"
    b = rt / "b"
    c = a / "c"
    d = a / "d"
    a.mkdir()
    b.touch()
    c.touch()
    d.mkdir()
    observ = list(iter_tree_braces(rt))
    tgt = list(
        Flattener()(
            [
                rt,
                b,
                [
                    a,
                    [
                        d,
                    ],
                    c,
                ],
            ]
        )
    )
    assert observ == tgt


def test_io_utils_basic(tmp_path_factory):
    """
    basic test for read/write functions & file factories for common file types
    """
    rt = tmp_path_factory.mktemp("io_basic")
    # npy
    arr = np.arange(10, dtype=np.int32)
    npf: File[np.ndarray] = io.NpyFile(rt / "f.npy")
    assert io.is_npy(npf)
    npf.write(arr)
    arr2 = npf.read()
    assert np.all(arr == arr2) and arr2.dtype == arr.dtype
    assert io.is_npy(npf)
    # TODO : for other types
    # yaml
    # toml
    # text
    # ?analyze75


def test_file_tables(tmp_path_factory):
    rt = tmp_path_factory.mktemp("ftables")
    dataset = {
        "a": "ab",
        "b": "bc",
        "c": "cd",
        "d": "de",
        "e": "ef",
    }
    psf = PrefixSuffixFormatter(rt, "_file.txt")
    ft = psf.to_FileTable(io.read_text, io.write_text)
    foreach(setitem(ft, k, v) for k, v in dataset.items())

    assert all(str(p).endswith(psf.suffix) for p in psf.prefix.iterdir())
    assert set(ft.keys()) == set(dataset.keys())
    dataset_reloaded = {k: ft[k] for k in dataset.keys()}
    assert dataset_reloaded == dataset
    # maybe add another setup?
    # rt = tmp_path_factory.mktemp("ftables2")
