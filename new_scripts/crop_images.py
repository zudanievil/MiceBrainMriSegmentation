import sys
import numpy as np
sys.path.append('..')
from new_lib.core import info_classes as ic
from new_lib.pipelines import crop_images


def images_to_numpy(ifi: ic.ImageFolderInfo):
    for i in ifi.raw_iter():
        img = i.raw_image()
        np.save(i.image_path(), img)


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        image_folder_info = ic.ImageFolderInfo(arg)
        images_to_numpy(image_folder_info)
        crop_images.main(image_folder_info)
