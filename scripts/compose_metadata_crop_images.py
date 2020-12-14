# /usr/bin/env python3

if __name__ == '__main__':
    import sys
    import pathlib

    root_dir = pathlib.Path(__file__).parent.parent
    sys.path.append(str(root_dir))

    from lib import load_config, project_ops
    load_config(project_ops)

    project = pathlib.Path('c:/users/user/desktop/new_segm/cgtg_c57bl')
    meta_source = project / 'pre_meta'

    project_ops.compose_metadata(project, meta_source)
    project_ops.crop_and_resize_images(project, save_png_previews=True)
