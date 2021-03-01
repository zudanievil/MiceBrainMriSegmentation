#! /usr/bin/env python3

import sys
import pathlib
import numpy as np
sys.path.append('..')
from lib import load_config, segmentation_by_atlas as sba, ontology as ont, segmentation_results_postprocessing as srp

if __name__ == '__main__':
    load_config(sba, ont)
    project = pathlib.Path(sys.argv[1])
    if len(sys.argv) == 3:
        batch_range = slice(*[int(a) for a in sys.argv[2].split(':')])
    else:
        batch_range = None
    spec_name = "pairwise"
    image_comparison_type = "pairwise"
    masks_folder = pathlib.Path('c:/users/user/files/lab/masks_actual')
    structure_names_for_significance_table = (
        'Main Olfactory Bulb',
        'Olfactory Nerve',
        'Anterior Olfactory Nucleus',
        'Piriform Area',
        'Piriform-Amygdalar Area',
        'Cortical Amygdalar Area',
        'Entorhinal Area',
        'Olfactory Tubercle',
        )

    print("project: ", project)
    print("spec_name: ", spec_name)
    print("batch_range: ", batch_range)
    input("press anything to continue")

    sba.segment_batches.call(project, spec_name=spec_name, masks_folder=masks_folder,
                             save_intersection_images=True, mask_permutation=lambda x: np.flip(x, axis=1),
                             batch_range=batch_range, comparison_type=image_comparison_type)
    sba.collect_segmentation_results.call(project, spec_name)
    sba.refactor_summary(project, spec_name)
    sba.plot_segmentation_results(project, spec_name)
    srp.make_kinetics_table(project, spec_name)
    sl = ont.get_substructure_list(masks_folder, structure_names_for_significance_table)
    srp.make_significance_table(project, spec_name, sl)

    print("""
    ========================
    ++      ALL DONE      ++
    ========================
    """)
