# /usr/bin/env python3

"""
render all the possible masks for the slice to use them later
this allows masks to be used by concurrent/parallel processes
takes about 5-10 min per slice
"""

if __name__ == '__main__':
    import sys
    import pathlib
    root_dir = pathlib.Path(__file__).parent.parent
    sys.path.append(str(root_dir))

    from lib import load_config, ontology, project_ops
    load_config(project_ops, ontology)

    masks_folder = pathlib.Path('C:/Users/user/Desktop/new_segm/masks_01')
    ontology.prerender_masks.call(masks_folder)
