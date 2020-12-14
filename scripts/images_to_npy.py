# /usr/bin/env python3

if __name__ == '__main__':
    import pathlib
    import numpy as np

    project = pathlib.Path('c:/users/user/desktop/new_segm/cgtg_abracadabra')

    image_shape = (256, 256)  # height, width
    image_dtype = np.int32
    for path in (project / 'img_raw').iterdir():
        name = path.name
        save_path = project / 'img' / name
        img = np.fromfile(path, dtype=image_dtype).reshape(image_shape)
        np.save(save_path, img, fix_imports=False)
