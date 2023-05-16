"""
Some shared functionality, factored out.
default handlers, pipeline errors, etc
"""

import matplotlib.pyplot as plt

from BrainS.prelude import *
from BrainS.lib.functional import ValDispatch
from BrainS.lib.config_utils import DocumentedNamespace as DNS

__all__ = [
    "DefaultHandlers",
    "PipelineErrors",
    "register_error",
    "explain_error",
    "ListErrorLog",
]

"""
this section may seem overly dynamic. usually, type fluency is a bad thing.
however, data-processing pipelines are very susceptible to changes and corrections,
so argument broadcasting, non-specific type definitions,
mutable dispatches and enums (argparse.Namespace) can be a good fit here.
they make things easier to hack from outside:
you do not need to change this module's code, to change the behaviour.
"""

DefaultHandlers = DNS(_name="Default Handlers")
DefaultHandlers._doc = """
This is a collection of default handler objects.
Handlers are, basically, callables,
that handle common effectful operations 
(exceptions, kv-storage, io, thread communication), 
separating them from the main logic.
This is more user-friendly than monadic
errors and more explicit than java-styled exceptions
""".strip()

PipelineErrors = DNS(_name="Pipeline Errors")
PipelineErrors._doc = """
a place to store error types (normally strings)
""".strip()


def register_error(name: str, doc: str = None):
    """
    register a new PipelineError.
    this allows to distribute the error definitions,
    yet collect all errors in a single object
    """
    setattr(PipelineErrors, name, name)
    if doc is not None:
        setattr(PipelineErrors, f"_{name}_doc", doc)


register_error(
    "DEFAULT",
    "simple error with a message.\nUse like: `Err(('DEFAULT', 'my message'))`",
)


@ValDispatch.new(
    PipelineErrors.DEFAULT, doc="explain one of the pipeline errors arguments"
)
def explain_error(_, x: str) -> str:
    """when string error is passed, return it"""
    return x


class ListErrorLog:
    __slots__ = "lst", "stream", "max_length"

    def __init__(self, stream: Opt[Any] = sys.stderr, max_length: int = 1000):
        self.lst = []
        self.stream = stream
        self.max_length = max_length

    def __call__(self, e: Err):
        self.lst.append(e)
        if len(self.lst) > self.max_length:
            self.lst.pop(0)
        if self.stream is not None:
            self.stream.write(explain_error(*e.data))

    def clear(self):
        self.lst.clear()


DefaultHandlers.soft_error = ListErrorLog()
DefaultHandlers._soft_error_type = Fn[[Err], None]
DefaultHandlers._soft_error_doc = """
used for handling soft (recoverable) errors
""".strip()

_fig_t = "matplotlib.figure.Figure"


def save_png_96_dpi(fig: _fig_t, fname: os.PathLike):
    fname = str(fname)
    if not fname.endswith(".png"):
        fname += ".png"
    fig.savefig(fname, dpi=96, format="png")
    plt.close(fig)


DefaultHandlers.flush_plot = save_png_96_dpi
DefaultHandlers._flush_plot_type = Fn[[_fig_t, os.PathLike], None]
DefaultHandlers._flush_plot_doc = """
used for get rid of the plot one way or another. 
normally, saves the plot and closes it.
""".strip()


class ProgressBar:
    """a utility wrapper around tqdm.tqdm"""

    __slots__ = "_tqdm"
    # I've decided to omit `__init__`, because it is not a part of this handler api.

    def update(
        self, add: int = 1, message: str = None, move_to: int = None
    ) -> None:
        """either add n points to the progress or move_to position. optionally set a message"""
        s = self._tqdm
        if s is None:
            return
        if message:
            s.set_description(message, refresh=False)
        if move_to is not None:
            s.n = move_to
            s.display()
        else:
            s.update(add)

    def __exit__(self, exc_type=None, exc_value=None, traceback=None) -> None:
        """close the bar"""
        s = self._tqdm
        if s is None:
            return
        s.__exit__(exc_type, exc_value, traceback)

    def __repr__(self):
        s = self._tqdm
        if s is None:
            return f"{no_ProgressBar.__name__}()"
        return (
            f"{new_ProgressBar.__name__}({s.total}) [progress={s.n}/{s.total}]"
        )

    close = __exit__
    __enter__ = identity


def new_ProgressBar(size: Opt[int] = None) -> ProgressBar:
    """make new progress bar"""
    from tqdm import tqdm

    s = ProgressBar()
    s._tqdm = tqdm(total=size)
    return s


def no_ProgressBar(*_, **__) -> ProgressBar:
    """make a phony progress bar (no side effect)"""
    s = ProgressBar()
    s._tqdm = None
    return s


DefaultHandlers.get_progress_bar = new_ProgressBar
DefaultHandlers._get_progress_bar_type = Fn[[int], ProgressBar]
DefaultHandlers._get_progress_bar_doc = """
construct simple class for progress/status reporting.
""".strip()
