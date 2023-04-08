class val_dispatch:
    """
    Typical in controllers and self-regulating programs. Example:
    ```
    PAUSE, RESUME = 1, 2
    @val_dispatch.new(PAUSE)
    def switch(_, channel: int):...
    @switch.register(RESUME)
    def switch(_, channel: int):...
    # more states can be added later
    # without changing the existing code
    def main():
        switch(RESUME, channel=5)
    ```
    Most useful when you have a huge
    and messy state transition graph --
    10+ possible states, and each transition
    is associated with its own function.
    """
    __slots__ = ("name", "registry")

    def __init__(self, name):
        self.name = name
        self.registry = dict()

    def register(self, val) -> _F[[callable], "val_dispatch"]:
        def clj(f: callable) -> val_dispatch:
            self.registry[val] = f
            return self
        return clj

    @classmethod
    def new(cls, val) -> _F[[callable], "val_dispatch"]:
        def clj(f: callable) -> val_dispatch:
            d = cls(f.__name__)
            return d.register(val)(f)
        return clj

    def dispatch(self, val) -> _F:
        return self.registry[val]

    def __call__(self, *args, **kwargs):
        return self.dispatch(args[0])(*args, **kwargs)

    def __repr__(self,) -> str:
        return f"val_dispatch {self.name}"