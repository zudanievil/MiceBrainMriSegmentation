
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append('..')
from lib import load_config, segmentation_by_atlas, ontology

if __name__ == '__main__':
    config_path = pathlib.Path('config_vs3h.yml')
    load_config(segmentation_by_atlas, ontology, config_path=config_path)
    masks_folder = pathlib.Path('c:/users/user/files/lab/masks_01')
    for exp in ('cgtg', 'c57bl'):
        project = pathlib.Path('C:\\Users\\user\\files\\lab\\CgTg\\') / exp
        segmentation_temp = project / '.segmentation_temp'
        flip_horizontally = lambda x: np.flip(x, axis=1)
        segmentation_by_atlas.segment_batches.call(project, masks_folder=masks_folder,
                save_intersection_images=True, mask_permutation=flip_horizontally)
        segmentation_by_atlas.collect_segmentation_results.call(project, result_file_name='partial_left.txt')
