"""
for consistent file io
"""
import os
from pathlib import Path
from typing import NewType

import toml as toml
import yaml as yaml
import numpy as np

from . import filesystem as fs
from .functional import Classifier
from ._read_analyze75 import read_analyze75, is_analyze75

__all__ = [
    "file",
    "read_toml",
    "write_toml",
    "is_toml",
    "TomlFile",
    "TomlDir",
    "read_yaml",
    "write_yaml",
    "is_yaml",
    "YamlFile",
    "YamlDir",
    "read_text",
    "write_text",
    "is_text",
    "TextFile",
    "read_npy",
    "write_npy",
    "is_npy",
    "NpyFile",
    "NpyDir",
    "read_analyze75",
    "is_analyze75",
    "Analyze75File",
    # "Analyze75Dir",
]


# <editor-fold descr="toml">
Toml = NewType("Toml", dict)


def read_toml(path) -> Toml:
    with open(path, "rt") as f:
        return toml.load(f)  # type: ignore


def is_toml(path) -> bool:
    p = Path(path)
    return (p.is_file() or not p.exists()) and p.suffix == ".toml"


def write_toml(path, value) -> None:
    with open(path, "wt") as f:
        toml.dump(value, f)


TomlFile: fs.FileFactory[Toml] = fs.FileFactory(read_toml, write_toml)


@Classifier.new(
    "try to determine file type and return a corresponding ``fs.File``"
)
def file(path) -> fs.File[Toml]:
    if is_toml(path):
        return TomlFile(path)


def TomlDir(
    prefix: os.PathLike, file_suffix: str = ".toml"
) -> fs.FileTable[str, Toml]:
    return fs.PrefixSuffixFormatter(Path(prefix), file_suffix).to_FileTable(
        read_toml, write_toml
    )


# </editor-fold>


# <editor-fold descr="yaml">
Yaml = NewType("Yaml", dict)


def read_yaml(path) -> Yaml:
    with open(path, "rt") as f:
        return yaml.safe_load(f)  # type: ignore


def is_yaml(path) -> bool:
    p = Path(path)
    return (p.is_file() or not p.exists()) and (
        p.suffix == ".yaml" or p.suffix == ".yml"
    )


def write_yaml(path, value) -> None:
    with open(path, "wt") as f:
        yaml.safe_dump(value)


YamlFile: fs.FileFactory[Yaml] = fs.FileFactory(read_yaml, write_yaml)


@file.add_arm()
def file(path) -> fs.File[Yaml]:
    if is_yaml(path):
        return YamlFile(path)


def YamlDir(
    prefix: os.PathLike, file_suffix: str = ".yml"
) -> fs.FileTable[str, Yaml]:
    return fs.PrefixSuffixFormatter(Path(prefix), file_suffix).to_FileTable(
        read_yaml, write_yaml
    )


# </editor-fold>


# <editor-fold descr="plain text">
def read_text(path) -> str:
    with open(path, "rt") as f:
        return f.read()


def is_text(path) -> bool:
    p = Path(path)
    return (p.is_file() or not p.exists()) and p.suffix == ".txt"


def write_text(path, value) -> None:
    with open(path, "wt") as f:
        f.write(value)


TextFile = fs.FileFactory(read_text, write_text)


@file.add_arm()
def file(path) -> fs.File[str]:
    if is_text(path):
        return TextFile(path)


# </editor-fold>


# <editor-fold descr="npy">
# Npy = NewType("Npy", np.ndarray)
def read_npy(path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def write_npy(path, value) -> None:
    np.save(path, value, allow_pickle=False)


def is_npy(path) -> bool:
    p = Path(path)
    return (p.is_file() or not p.exists()) and p.suffix == ".npy"


NpyFile: fs.FileFactory[np.ndarray] = fs.FileFactory(read_npy, write_npy)


def NpyDir(
    prefix: os.PathLike, file_suffix: str = ".npy"
) -> fs.FileTable[str, np.ndarray]:
    return fs.PrefixSuffixFormatter(Path(prefix), file_suffix).to_FileTable(
        read_npy, write_npy
    )


@file.add_arm()
def file(path) -> fs.File[np.ndarray]:
    if is_npy(path):
        return NpyFile(path)


# </editor-fold>


# <editor-fold descr="analyze">
Analyze75 = NewType("Analyze75", np.ndarray)


Analyze75File: fs.FileFactory[Analyze75] = fs.FileFactory(read_analyze75)  # type: ignore


@file.add_arm()
def file(path) -> fs.File[Analyze75]:
    if is_analyze75(path):
        return Analyze75File(path)


# </editor-fold>
