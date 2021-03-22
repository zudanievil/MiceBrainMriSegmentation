import sys
import pathlib
import numpy as np
sys.path.append('..')
from ignore_lib import load_config, segmentation_by_atlas

if __name__ == '__main__':
    args = [pathlib.Path(a) for a in sys.argv[1:]]
    load_config(segmentation_by_atlas, config_path=args[0])
    print(args)
    input('press anything to continue')
    for project in args[1:]:
        segmentation_temp = project / '.segmentation_temp'
        # segmentation_temp.mkdir()
        segmentation_by_atlas.find_batches_for_segmentation.call(project)
        print('done')
