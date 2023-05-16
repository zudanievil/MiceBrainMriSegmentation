# import numpy as np
import pandas as pd

from ..prelude import *
from ..lib.math import sum_mean_std

# from ._share import *


class aggregate_segmentation_results(NamedTuple):
    idx_cols: List[str] = ["structure", "group", "hour", "animal"]
    reduce_cols: List[str] = ["frame"]
    structures: List[str] = []
    pval_thresholds: List[float] = [1, 0.05, 0.01]

    @staticmethod
    def load_group(path, meta: Dict[str, Any] = None) -> pd.DataFrame:
        t = pd.read_csv(path, sep="\t")
        if meta is not None:
            for k, v in meta.items():
                t[k] = v
        return t

    def process_group(self, t: pd.DataFrame) -> pd.DataFrame:
        all_cols = [c for c in t.columns if c.startswith(("mean", "std", "px"))]
        all_cols += self.idx_cols + self.reduce_cols
        t = t[all_cols]  # shallow copy

        # filter
        if self.structures:
            t.set_index(["structure"], inplace=True)
            t = t.loc[self.structures]
            t.reset_index(inplace=True)

        t.set_index(self.idx_cols + self.reduce_cols, inplace=True)
        if not self.reduce_cols:
            return t

        # group_by + sum
        colnames = []
        accum = []
        for p in self.pval_thresholds:
            cols = [f"mean (p <{p})", f"std (p <{p})", f"px (p <{p})"]
            mean_std_count = sum_mean_std(
                mean=t[cols[0]],
                std=t[cols[1]],
                count=t[cols[2]],
                group_reduce=lambda series: series.groupby(
                    level=self.idx_cols
                ).sum(min_count=1),
            )
            accum.extend(mean_std_count)
            colnames.extend(cols)
        t = pd.concat(accum, axis=1)
        t.columns = colnames
        return t

    def __call__(
        self,
        paths_n_metas: Iterable[
            Union[os.PathLike, Tuple[os.PathLike, Opt[Dict[str, Any]]]]
        ],
        save_to,
    ):
        """example signature `self(["res1.txt", ("res_wo_group.txt", {"group": 3}), ("res2.txt", None),],
        save_to="agg.txt")`"""
        chunks = [
            self.load_group(*sd)
            if isinstance(sd, tuple)
            else self.load_group(sd)
            for sd in paths_n_metas
        ]
        chunks = [self.process_group(c) for c in chunks]
        table = pd.concat(chunks, axis=0)
        table.to_csv(save_to, sep="\t")
