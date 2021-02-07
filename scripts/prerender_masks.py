# /usr/bin/env python3

"""
render all the possible masks for the slice to use them later
this allows masks to be used by concurrent/parallel processes
takes about 5-10 min per slice
"""
import pathlib
import sys

sys.path.append('..')
from lib import load_config, ontology, project_ops

if __name__ == '__main__':
    load_config(project_ops, ontology)
    masks_folder = pathlib.Path(sys.argv[1])
    ontology.prerender_masks.call(masks_folder)
