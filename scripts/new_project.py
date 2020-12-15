# /usr/bin/env python3

if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.append('..')

    from lib import load_config, project_ops
    load_config(project_ops)

    project = pathlib.Path('c:/users/user/desktop/new_segm/example_project')
    project_ops.new_project_folder(project)
