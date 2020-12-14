# /usr/bin/env python3

"""
calculate statistics over masks (left half of the brain)
can run in parallel with anoter such process,
because it only reads shared data (brain structure masks)
"""
if __name__ == '__main__':
    import sys
    import pathlib
    root_dir = pathlib.Path(__file__).parent.parent
    sys.path.append(str(root_dir))

    import numpy as np
    from lib import load_config, segmentation_by_atlas
    load_config(segmentation_by_atlas)

    project = pathlib.Path('c:/users/user/desktop/new_segm/cgtg_cgtg')
    masks_folder = pathlib.Path('c:/users/user/desktop/new_segm/masks_01')

    segmentation_temp = project / '.segmentation_temp'
    segmentation_temp.mkdir()

    segmentation_by_atlas.find_batches_for_segmentation.call(project)
    flip_horizontally = lambda x: np.flip(x, axis=1)  # svgs in that atlas consist of right brain side only
    segmentation_by_atlas.segment_batches.call(project, masks_folder=masks_folder,
                save_intersection_images=True, mask_permutation=flip_horizontally)
    segmentation_by_atlas.collect_segmentation_results.call(project, result_file_name='results_left.txt')

