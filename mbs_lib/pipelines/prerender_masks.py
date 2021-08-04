import sys
import pathlib
import tqdm
import typing
import subprocess
import PIL.Image
import numpy
import datetime
from xml.etree import ElementTree
from ..core import info_classes
from ..utils import miscellaneous_utils


shape_dict_type = typing.Dict[str, typing.Tuple[int, int]]
maybe_string_collection = typing.Union[typing.Iterable[str], None]


class EmptyMaskException(Exception):
    pass


def main(ontology_folder_info: info_classes.ontology_folder_info_like,
         image_folder_info: info_classes.image_folder_info_like,
         frames: typing.List[str] = None):
    ontology_folder_info = info_classes.OntologyFolderInfo(ontology_folder_info)
    image_folder_info = info_classes.ImageFolderInfo(image_folder_info)
    frame_shapes = get_frame_shapes_dict(frames, ontology_folder_info, image_folder_info)
    spec = ontology_folder_info.specification()['rendering_constants']
    assert_no_collisions(ontology_folder_info, frame_shapes)
    for frame, frame_shape in frame_shapes.items():
        ontology_info = ontology_folder_info.ontology_info(frame)
        structure_tree_root = ontology_info.default_tree().getroot()
        structures = tuple(structure_tree_root.iter('structure'))
        previous_inspected_structure_level = 0
        previous_structure_not_found = False
        progress_bar = tqdm.tqdm(leave=False, total=len(structures), file=sys.stdout)
        progress_bar.set_description_str(f'rendering for {frame}')
        for structure in structures:
            progress_bar.set_postfix_str(structure.attrib['name'])
            progress_bar.update()
            structure_level = int(structure.attrib['level'])
            if previous_structure_not_found and previous_inspected_structure_level < structure_level:
                continue
            try:
                svg_source_path = ontology_info.svg_path()
                structure_parents = miscellaneous_utils.get_structure_parents(structure_tree_root, structure.attrib['name'])
                mask_path_rel, mask_path_abs = compose_structure_path(ontology_info, structure_parents)
                mask_path_abs.parent.mkdir(exist_ok=True, parents=True)
                render_structure_mask(svg_source_path, mask_path_abs, structure, frame_shape, spec)
                structure.attrib['filename'] = mask_path_rel.as_posix()
                previous_structure_not_found = False
            except EmptyMaskException:
                structure_parents[-2].remove(structure)  # -1 is the structure itself
                maybe_remove_residual_paths(mask_path_abs)
                previous_structure_not_found = True
            previous_inspected_structure_level = structure_level
        print(f"\nTotal found structures {len(tuple(structure_tree_root.iter('structure')))}")
        progress_bar.close()
        save_structure_tree(structure_tree_root, ontology_info, spec)


def get_frame_shapes_dict(frames: maybe_string_collection,
                          ontology_folder_info: info_classes.OntologyFolderInfo,
                          image_folder_info: info_classes.ImageFolderInfo) -> shape_dict_type:
    available_frames = set(ontology_folder_info.frames())
    if frames is not None:
        frames = set(frames)
        assert frames.issubset(available_frames), \
            f'not all of {frames} have a corresponding svg file'
    else:
        frames = available_frames
    shapes = image_folder_info.specification()['cropped_image_shapes']
    assert frames.issubset(shapes), \
        f'image_folder specification does not include shapes for all of {frames}'
    return {f: sh for f, sh in shapes.items() if f in frames}


def assert_no_collisions(ontology_folder_info: info_classes.OntologyFolderInfo,
                         frame_shapes: shape_dict_type):
    folder = ontology_folder_info.folder()
    collisions = []
    for frame in frame_shapes:
        if (folder / frame).exists():
            collisions.append(frame)
    if collisions:
        raise AssertionError(f'{folder} contains names: {collisions} that '
                             f'will collide with the new subfolders with masks')


def compose_structure_path(ontology_info: info_classes.OntologyInfo,
                           structure_parents: typing.List[ElementTree.Element]) -> (pathlib.Path, pathlib.Path):
    acronyms = [s.attrib['acronym'] for s in structure_parents]
    rel_p = pathlib.Path('/'.join(acronyms)) / structure_parents[-1].attrib['name']
    abs_p = (ontology_info.masks_folder() / rel_p).with_suffix('.png')
    return rel_p, abs_p


def maybe_remove_residual_paths(png_mask_path: pathlib.Path):
    paths = (
        png_mask_path,
        png_mask_path.with_suffix('.svg'),
        png_mask_path.parent
    )
    for p in paths:
        if p.exists() and p.is_dir():
            p.rmdir()
        elif p.exists():
            p.unlink()


def save_structure_tree(structure_tree_root: ElementTree.Element,
                        ontology_info: info_classes.OntologyInfo, spec: dict):
    p = ontology_info.tree_path()
    metadata = ElementTree.Element('rendering_info')
    metadata.attrib = {k: str(v) for k, v in spec.items()}
    metadata.attrib['time'] = str(datetime.datetime.now())
    structure_tree_root.append(metadata)
    ElementTree.ElementTree(structure_tree_root).write(p,  encoding='utf-8')

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# mask rendering functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def render_structure_mask(svg_source_path: pathlib.Path, png_mask_path: pathlib.Path,
                          structure: ElementTree.Element, frame_shape: (int, int), spec: dict):
    ids = get_structure_ids(structure, spec)
    svg_mask_path = png_mask_path.with_suffix('.svg')
    svg_mask_from_ids(svg_source_path, svg_mask_path, ids, spec)
    svg_mask_to_png(svg_mask_path, png_mask_path, frame_shape, spec)
    png_to_grayscale_png(png_mask_path, png_mask_path, spec)
    svg_mask_path.unlink()


def get_structure_ids(structure: ElementTree.Element, spec: dict) -> typing.Set[str]:
    if spec['include_substructures']:
        ids = []
        for substructure in structure.iter('structure'):
            ids.append(substructure.attrib['id'])
    else:
        ids = [structure.attrib['id']]
    return set(ids)


def svg_mask_from_ids(svg_path: pathlib.Path, save_path: pathlib.Path, ids: typing.Set[str], spec: dict):
    svg_root = ElementTree.parse(svg_path).getroot()
    empty = True
    for node in tuple(svg_root.iter()):  # tree iterator misbehaves if you modify the tree inside the loop
        if node.tag.endswith('namedview'):
            node.set('pagecolor', "#000000")
        try:
            if node.attrib['structure_id'] in ids:
                node.set('style', 'stroke:none;fill:#ffffff;fill-opacity:1')
                empty = False
            elif node.attrib['structure_id'] == spec['svg_crop_id']:
                node.set('style', 'stroke:none;fill:#000000;fill-opacity:0')
            else:
                miscellaneous_utils.remove_node_xml_from_the_tree(svg_root, node)
        except KeyError:
            if not node.tag.endswith(tuple(spec['allowed_svg_tag_tails'])):
                miscellaneous_utils.remove_node_xml_from_the_tree(svg_root, node)
    if empty:
        raise EmptyMaskException()
    ElementTree.ElementTree(svg_root).write(save_path,  encoding='utf-8')


def svg_mask_to_png(load_path: pathlib.Path, save_path: pathlib.Path, shape: (int, int), spec: dict):
    cmd = spec['rendering_command'].format(
        inkscape=spec['inkscape_executable_path'], export_id=spec['svg_crop_id'],
        src_path=load_path, dst_path=save_path, height=shape[0], width=shape[1])
    for i in range(3):
        try:  # after inkscape 1.0 update, sometimes process does not exit,
            # although render appears to be normal, it's good to be sure that it's complete
            process = subprocess.Popen(cmd)
            process.wait(10)
            process.kill()
            break
        except subprocess.TimeoutExpired as e:
            print('Inkscape process timeout', load_path, file=sys.stderr)
            process.kill()
            if i >= 3:
                print('Too much timeouts', load_path, file=sys.stderr)
                raise e


def png_to_grayscale_png(load_path: pathlib.Path, save_path: pathlib.Path,
                         spec: dict, check_mask_size: bool = True):
    img = PIL.Image.open(load_path)
    img.load()
    img = img.convert('L')
    if check_mask_size:
        mask = numpy.array(img.getdata(), dtype=numpy.uint8).reshape((img.size[1], img.size[0]))
        if (mask > 127).sum() < spec['min_mask_size']:
            raise EmptyMaskException
    img.save(save_path)
