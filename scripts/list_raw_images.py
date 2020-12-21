# /usr/bin/env python3

if __name__ == '__main__':
    import pathlib

    project = pathlib.Path('c:/users/user/desktop/new_segm/cgtg_cgtg')

    with (project / 'ij_img_list.txt').open('wt') as f:
        for path in (project / 'img_raw').iterdir():
            print(path.name, file=f)
    with (project / 'ij_pointer.txt').open('wt') as f:
        print(0, file=f)
