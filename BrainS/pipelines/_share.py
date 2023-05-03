"""
Some shared functionality, factored out.
default handlers, pipeline errors, etc
"""

from argparse import Namespace as NS
import matplotlib.pyplot as plt

from BrainS.prelude import *
from BrainS.lib.functional import ValDispatch


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

DefaultHandlers = NS()
DefaultHandlers.doc = """
This is a collection of default handler objects.
Handlers are, basically, callables,
that handle common effectful operations 
(exceptions, kv-storage, io, thread communication), 
separating them from the main logic.
This is more user-friendly than monadic
errors and more explicit than java-styled exceptions
"""

PipelineErrors = NS()


def register_error(name: str):
    """
    register a new PipelineError.
    this allows to distribute the error definitions,
    yet collect all errors in a single object
    """
    setattr(PipelineErrors, name, name)


@ValDispatch.new(None)
def explain_error(*_) -> str:
    """explain one of the pipeline errors arguments"""
    return "no error"


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
DefaultHandlers.soft_error_type = Fn[[Err], None]
DefaultHandlers.soft_error_doc = """
used for handling soft (recoverable errors)
"""


def save_png_96_dpi(fig: "Figure", fname: os.PathLike):
    fname = str(fname)
    if not fname.endswith(".png"):
        fname += ".png"
    fig.savefig(fname, dpi=96, format="png")
    plt.close(fig)


DefaultHandlers.flush_plot = save_png_96_dpi
DefaultHandlers.flush_plot_type = Fn[["Figure", os.PathLike], None]
DefaultHandlers.flush_plot_doc = """
used for get rid of the plot one way or another. 
normally, saves the plot and closes it.
"""