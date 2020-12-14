# /usr/bin/env python3

if __name__ == '__main__':
    import sys
    import pathlib
    root_dir = pathlib.Path(__file__).parent.parent
    sys.path.append(str(root_dir))

    from lib import load_config, project_ops
    load_config(project_ops)

    project = pathlib.Path('c:/users/user/desktop/new_segm/example_project')
    project_ops.new_project_folder(project)
