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
    """
    This pipeline constructs grayscale png masks
    from Allen Brain Institute atlas sections.
    :param image_folder_info is needed for `frame_shapes` entry of the image folder configuration.
    :param frames: if not None, masks will be rendered for specified frames (sections) only. This
    allows parallelization of rendering process.
    """
    ontology_folder_info = info_classes.OntologyFolderInfo(ontology_folder_info)
    image_folder_info = info_classes.ImageFolderInfo(image_folder_info)
    frame_shapes = get_frame_shapes_dict(frames, ontology_folder_info, image_folder_info)
    spec = ontology_folder_info.configuration()['rendering_constants']
    check_inkscape_version(spec)
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


def get_inkscape_version(inkscape_executable: str) -> str:
    with subprocess.Popen(
            (inkscape_executable + " --version").split(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        out, err = p.communicate()
    return out.decode().split()[1]


def version_is_greater_equal(current: str, required: str) -> bool:
    ge = True
    for cv, rv in zip(current.split("."), required.split(".")):
        cv, rv = int(cv), int(rv)
        if cv > rv:
            ge = True
            break
        elif cv < rv:
            ge = False
            break
    return ge


def check_inkscape_version(spec: dict) -> None:
    required: str = spec["inkscape_minimal_version"]
    current = get_inkscape_version(spec["inkscape_executable_path"])
    assert version_is_greater_equal(current, required), f"Inkscape version is {current}, required {required}"


def get_frame_shapes_dict(frames: maybe_string_collection,
                          ontology_folder_info: info_classes.OntologyFolderInfo,
                          image_folder_info: info_classes.ImageFolderInfo) -> shape_dict_type:
    """loads shape dict, but only for the specified frames"""
    available_frames = set(ontology_folder_info.frames())
    if frames is not None:
        frames = set(frames)
        assert frames.issubset(available_frames), \
            f'not all of {frames} have a corresponding svg file'
    else:
        frames = available_frames
    shapes = image_folder_info.configuration()['cropped_image_shapes']
    assert frames.issubset(shapes), \
        f'image_folder configuration does not include shapes for all of {frames}'
    return {f: sh for f, sh in shapes.items() if f in frames}


def assert_no_collisions(ontology_folder_info: info_classes.OntologyFolderInfo,
                         frame_shapes: shape_dict_type):
    """checks that no prerendered structure masks exist for sections in frame shapes"""
    folder = ontology_folder_info.folder()
    collisions = []
    for frame in frame_shapes:
        if (folder / frame).exists():
            collisions.append(frame)
    if collisions:
        raise AssertionError(
            f'{folder} contains names: {collisions} that will collide with the new subfolders with masks')


def compose_structure_path(ontology_info: info_classes.OntologyInfo,
                           structure_parents: typing.List[ElementTree.Element]) -> (pathlib.Path, pathlib.Path):
    """makes path for structure.
    :returns path relative to atlas section folder; absolute path"""
    acronyms = [s.attrib['acronym'] for s in structure_parents]
    rel_p = pathlib.Path('/'.join(acronyms)) / structure_parents[-1].attrib['name']
    abs_p = (ontology_info.masks_folder() / rel_p).with_suffix('.png')
    return rel_p, abs_p


def maybe_remove_residual_paths(png_mask_path: pathlib.Path):
    """
    removes png_mask_path, corresponding .svg temp file and folder they are in.
    if some paths do not exist, skips them without any errors
    """
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
    """function that does whole rendering process"""
    svg_mask_path = png_mask_path.with_suffix('.svg')
    ids = get_structure_ids(structure, spec['include_substructures'])
    svg_mask_from_ids(svg_source_path, svg_mask_path, ids, spec['svg_crop_id'], tuple(spec['allowed_svg_tag_tails']))
    svg_mask_to_png(
        svg_mask_path, png_mask_path, frame_shape,
        spec['inkscape_executable_path'], spec['svg_crop_id'], spec['rendering_command'],)
    png_to_grayscale_png(png_mask_path, png_mask_path, spec['min_mask_size'])
    svg_mask_path.unlink()


def get_structure_ids(structure: ElementTree.Element, include_substructures: bool) -> typing.Set[str]:
    """gets ids for structures from the tree node"""
    if include_substructures:
        ids = []
        for substructure in structure.iter('structure'):
            ids.append(substructure.attrib['id'])
    else:
        ids = [structure.attrib['id']]
    return set(ids)


def svg_mask_from_ids(
        svg_path: pathlib.Path,
        save_path: pathlib.Path,
        ids: typing.Set[str],
        crop_id: str,
        allowed_tag_tails: typing.Tuple[str],
        ) -> None:
    """
    Produces temporary svg file (from a copy of atlas section) with only relevant parts left.
    Irrelevant svg objects are removed, background is set to black, remaining objects -- to white.
    :param ids: ids of structures that will be included
    :param crop_id: id of svg object that will serve as a bounding box (usually it's a rectangle)
    :param allowed_tag_tails: tuple of xml tag ends, that have no graphical representation
    """
    svg_root = ElementTree.parse(svg_path).getroot()
    empty = True
    for node in tuple(svg_root.iter()):  # tree iterator misbehaves if you modify the tree inside the loop
        if node.tag.endswith('namedview'):
            node.set('pagecolor', "#000000")
        try:
            if node.attrib['structure_id'] in ids:
                node.set('style', 'stroke:none;fill:#ffffff;fill-opacity:1')
                empty = False
            elif node.attrib['structure_id'] == crop_id:
                node.set('style', 'stroke:none;fill:#000000;fill-opacity:0')
            else:
                miscellaneous_utils.remove_node_xml_from_the_tree(svg_root, node)
        except KeyError:
            if not node.tag.endswith(allowed_tag_tails):
                miscellaneous_utils.remove_node_xml_from_the_tree(svg_root, node)
    if empty:
        raise EmptyMaskException()
    ElementTree.ElementTree(svg_root).write(save_path,  encoding='utf-8')


def svg_mask_to_png(
        load_path: pathlib.Path,
        save_path: pathlib.Path,
        shape: (int, int),
        inkscape_exe: str,
        crop_id: str,
        rendering_command: str,
        ) -> None:
    """
    Uses Inkscape CLI to render svg. This produces an RGB png image.
    """
    cmd = rendering_command.format(
        inkscape=inkscape_exe, export_id=crop_id,
        src_path=load_path, dst_path=save_path, height=shape[0], width=shape[1])
    for i in range(3):  # in case rendering fails because of some internal error (had this problem several times)
        try:
            process = subprocess.Popen(cmd.split())
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
                         min_mask_size: typing.Optional[int] = None):
    """
    Converts a normal png to grayscale, because we need 1 channel only.
    Checks if mask has enough pixels
    """
    img = PIL.Image.open(load_path)
    img.load()
    img = img.convert('L')
    if min_mask_size is not None:
        mask = numpy.array(img.getdata(), dtype=numpy.uint8).reshape((img.size[1], img.size[0]))
        if (mask > 127).sum() < min_mask_size:
            raise EmptyMaskException
    img.save(save_path)
