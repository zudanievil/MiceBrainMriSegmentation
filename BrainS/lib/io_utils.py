"""
for consistent file io
"""
import os

import toml as toml
import yaml as yaml
import numpy as np

from . import filesystem as fs

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
    # "read_analyze75",
    # "Analyze75File",
]


def read_toml(path) -> dict:
    with open(path, "rt") as f:
        return toml.load(f)


def write_toml(path, value) -> None:
    with open(path, "wt") as f:
        toml.dump(value, f)


TomlFile = fs.FileFactory(read_toml, write_toml)


def read_yaml(path) -> dict:
    with open(path, "rt") as f:
        return yaml.safe_load(f)


def write_yaml(path, value) -> None:
    with open(path, "wt") as f:
        yaml.safe_dump(value)


YamlFile = fs.FileFactory(read_yaml, write_yaml)


def read_text(path) -> str:
    with open(path, "rt") as f:
        return f.read()


def write_text(path, value) -> None:
    with open(path, "wt") as f:
        f.write(value)


TextFile = fs.FileFactory(read_text, write_text)


def read_npy(path) -> np.ndarray:
    return np.load(os.fspath(path), allow_pickle=False)


def write_npy(path, value) -> None:
    np.save(os.fspath(path), value, allow_pickle=False)


NpyFile: fs.FileFactory[np.ndarray] = fs.FileFactory(read_npy, write_npy)


def read_analyze75(path) -> np.ndarray:
    ...


Analyze75File = ...  # todo
