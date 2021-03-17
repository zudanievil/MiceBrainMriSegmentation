if __name__ == '__main__':
    import sys
    import pathlib

    sys.path.append('..')
    from new_lib.core import info_classes as ic

    p = pathlib.Path('C:/Users/user/files/lab')
    groups = ('c57bl_8w', 'cgtg_8w')

    for g in groups:
        image_folder_info = ic.ImageFolderInfo(p / g)
        image_folder_info.write()
