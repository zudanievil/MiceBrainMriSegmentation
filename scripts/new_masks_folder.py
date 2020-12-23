# /usr/bin/env python3

"""
Sets up new folder for masks.
The 'new_masks_folder' function makes {masks_folder}/download_info.yml file.
all the other functions in this script rely on that file (download_slice_ids_table,
download_default_ontology, download_slice_svgs): given the same masks_info.yml and same lib config, they will
produce same files.
The download_info.yml file by default contains some example info and requires manual editing.
The idea of this script is, that you will run it first time (better with 'download_slice_svgs' call commented out,
since download_info.yml contains example svg downloading info, that you most certainly want to replace),
edit download_info.yml and then run script again with 'new_masks_folder' function commented out.
"""

if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.append('..')

    from lib import load_config, project_ops
    load_config(project_ops)

    masks_folder = pathlib.Path('c:/users/user/desktop/new_segm/example_masks_folder')
    project_ops.new_masks_folder(masks_folder)
    project_ops.download_slice_ids_table(masks_folder)
    project_ops.download_default_ontology(masks_folder)
    project_ops.download_slice_svgs(masks_folder)
