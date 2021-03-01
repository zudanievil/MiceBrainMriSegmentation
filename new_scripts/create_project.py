if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.append('..')
    from new_lib.core import info_classes as ic
    p = pathlib.Path(sys.argv[1])
    m = sys.argv[2] if len(sys.argv) > 2 else None
    proj = ic.ProjectInfo.new_single_root_project(p, m)
    proj.write_map(exist_ok=False)
    proj.create_folders(exist_ok=True)
