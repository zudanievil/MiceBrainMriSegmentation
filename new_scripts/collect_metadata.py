import pathlib
import sys
sys.path.append('..')
from new_lib.core import info_classes as ic
from new_lib.pipelines import collect_image_metadata


if __name__ == '__main__':
    not_found = []
    for arg in sys.argv[1:]:
        image_folder_info = ic.ImageFolderInfo(arg)
        not_found.extend(
            collect_image_metadata.main(image_folder_info))
    if not_found:
        raise AssertionError(f'\nThe following files were not found:\n{not_found},\n'
                             f'please fix this and restart the script')
