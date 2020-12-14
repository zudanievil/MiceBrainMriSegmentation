# /usr/bin/env python3

if __name__ == '__main__':
    import sys
    import pathlib
    root_dir = pathlib.Path(__file__).parent.parent
    sys.path.append(str(root_dir))

    from lib import load_config, project_ops
    load_config(project_ops)

    masks_folder = pathlib.Path('c:/users/user/desktop/new_segm/example_masks_folder')
    project_ops.new_masks_folder(masks_folder)
    project_ops.fetch_slice_ids_table(masks_folder / 'slice_ids.txt')