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
from ._read_analyze75 import read_analyze75

__all__ = [
    "read_toml",
    "write_toml",
    "TomlFile",
    "read_yaml",
    "write_yaml",
    "YamlFile",
    "read_text",
    "write_text",
    "TextFile",
    "read_npy",
    "write_npy",
    "NpyFile",
    "read_analyze75",
    "Analyze75File",
    "TomlDir",
    "YamlDir",
    "NpyDir",
]

Toml = NewType("Toml", dict)
Yaml = NewType("Yaml", dict)
# Npy = NewType("Npy", np.ndarray)


def read_toml(path) -> dict:
    with open(path, "rt") as f:
        return toml.load(f)


def write_toml(path, value) -> None:
    with open(path, "wt") as f:
        toml.dump(value, f)


TomlFile: fs.FileFactory[Toml] = fs.FileFactory(read_toml, write_toml)


def read_yaml(path) -> dict:
    with open(path, "rt") as f:
        return yaml.safe_load(f)


def write_yaml(path, value) -> None:
    with open(path, "wt") as f:
        yaml.safe_dump(value)


YamlFile: fs.FileFactory[Yaml] = fs.FileFactory(read_yaml, write_yaml)


def read_text(path) -> str:
    with open(path, "rt") as f:
        return f.read()


def write_text(path, value) -> None:
    with open(path, "wt") as f:
        f.write(value)


TextFile = fs.FileFactory(read_text, write_text)


def read_npy(path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def write_npy(path, value) -> None:
    np.save(path, value, allow_pickle=False)


NpyFile: fs.FileFactory[np.ndarray] = fs.FileFactory(read_npy, write_npy)

Analyze75File = fs.FileFactory(read_analyze75)  # type: ignore


def NpyDir(
    prefix: os.PathLike, file_suffix: str = ".npy"
) -> fs.FileTable[str, np.ndarray]:
    return fs.PrefixSuffixFormatter(Path(prefix), file_suffix).to_FileTable(
        read_npy, write_npy
    )


def YamlDir(
    prefix: os.PathLike, file_suffix: str = ".yml"
) -> fs.FileTable[str, Yaml]:
    return fs.PrefixSuffixFormatter(Path(prefix), file_suffix).to_FileTable(
        read_yaml, write_yaml
    )


def TomlDir(
    prefix: os.PathLike, file_suffix: str = ".toml"
) -> fs.FileTable[str, Toml]:
    return fs.PrefixSuffixFormatter(Path(prefix), file_suffix).to_FileTable(
        read_toml, write_toml
    )
