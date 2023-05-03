import pytest

from BrainS.prelude import *
from BrainS.lib.iterators import Flattener, foreach
from BrainS.lib.filesystem import (
    iter_tree_braces,
    File,
    FileTable,
    PrefixSuffixFormatter,
    repr_DynamicDirInfo,
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


@pytest.mark.visual
def test_repr_DynamicDirInfo(tmp_path_factory):
    class ReportDirInfo:
        """just some class that I've made up on spot"""

        def __init__(s, root: Path):
            s.root = root
            s.__root = "project dir root"
            s.plots_trouble = root / "plots/tsh"
            s.__plots_trouble = "plots for troubleshooting"
            s.run_cfg = io.TomlFile(root / "run.toml")
            s.__run_cfg = "configuration for the pipeline"
            s.checkpoint_files = FileTable(
                lambda x: root / (x + ".ckpt"),
                _read=io.read_text,
                _write=io.write_text,
            )

        __repr__ = repr_DynamicDirInfo

    root = tmp_path_factory.mktemp("file_collections")

    report_dir = ReportDirInfo(root)

    report_dir.plots_trouble.mkdir(exist_ok=True, parents=True)
    report_dir.run_cfg.path.touch()
    print(report_dir)
