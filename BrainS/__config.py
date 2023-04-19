# prefer using via prelude
# prefer assigning values at runtime rather than setting them here
from pathlib import Path as _Path


resource_dir = (_Path(__file__) / "../../resources").resolve()
"""
where default configuration files, etc are stored
"""
inkscape_cmd = "inkscape"
"""
for using with inkscape cli. check if it works with `$inkscape_cmd --version`
"""
about = "https://github.com/zudanievil/MiceBrainMriSegmentation/tree/v3"


__version__ = "set from _version.py at runtime"
version_tuple = "set from _version.py at runtime"

