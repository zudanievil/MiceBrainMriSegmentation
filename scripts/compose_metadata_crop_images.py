# /usr/bin/env python3

if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.append('..')

    from lib import load_config, project_ops
    load_config(project_ops)

    project = pathlib.Path('c:/users/user/desktop/new_segm/cgtg_c57bl')
    meta_source = project / 'pre_meta'

    project_ops.compose_metadata(project, meta_source)
    project_ops.crop_and_resize_images(project, save_png_previews=True)
