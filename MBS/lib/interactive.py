from .. prelude import *

@do_it
def _IPYTHON():
    try:
        return get_ipython()
    except NameError:
        return None


if _IPYTHON is not None:
    pass
