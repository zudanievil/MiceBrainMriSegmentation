# /usr/bin/env python3

"""
plot summary
"""

if __name__ == '__main__':
    import sys
    import pathlib
    root_dir = pathlib.Path(__file__).parent.parent
    sys.path.append('..')

    from ignore_lib import load_config, segmentation_by_atlas
    load_config(segmentation_by_atlas)

    projects = pathlib.Path('c:/users/user/desktop/new_segm/')
    for p in ['cgtg_cgtg', 'cgtg_c57bl']:
        project = projects / p
        segmentation_by_atlas.refactor_summary(project/'results_left.txt', project/'results_left_refactored.txt')
        segmentation_by_atlas.plot_segmentation_results(project, project/'results_left_refactored.txt')
