import urllib.request
import pandas
import numpy
from xml.etree import ElementTree
from ..core import info_classes

# todo: annotate


def execute_all(ontology_folder_info: info_classes.ontology_folder_info_like):
    """
    this functions sequentially calls other functions from this module:
    1.`download_slice_ids_table`
    2.`download_default_ontology`
    3.`download_svgs`
    see their docs for details
    """
    download_slice_ids_table(ontology_folder_info)
    download_default_ontology(ontology_folder_info)
    download_svgs(ontology_folder_info)
   
 
def download_slice_ids_table(ontology_folder_info: info_classes.ontology_folder_info_like) -> None:
    """
    downloads the list of atlas sections from Allen Brain Institute brain atlases.
    sections are annotated with linear coordinate range, specified by
    `downloading_arguments/slice_coordinates` keys in the ontology folder configuration.
    writes section ids and coordinates to a slice_ids.txt text table.
    """
    ontology_folder_info = info_classes.OntologyFolderInfo(ontology_folder_info)
    save_path = ontology_folder_info.folder() / 'slice_ids.txt'

    spec = ontology_folder_info.configuration()
    kwargs = spec['downloading_arguments']
    url = spec['downloading_urls']['slice_ids']
    url = url.format(**kwargs)
    urllib.request.urlretrieve(url, save_path)
    # xml to list
    root = ElementTree.parse(save_path).getroot()
    slice_ids = [int(node.text) for node in root.iter(tag="id")]
    del root
    slice_ids.reverse()
    # pair with coordinates
    slice_ids = numpy.array(slice_ids)
    coords = numpy.linspace(**kwargs['slice_coordinates'], num=len(slice_ids))
    t = pandas.DataFrame()
    t['ids'] = slice_ids
    t['coordinates'] = coords

    t.to_csv(save_path, sep='\t', index=False)
    print(f"saved section ids to {save_path}")


def download_default_ontology(ontology_folder_info: info_classes.ontology_folder_info_like) -> None:
    """
    downloads ontology (brain structure hierarchy tree and structure ids)
    from Allen Brain Institute brain atlases.
    refactors it to make it more human-friendly (still xml, though)
    """
    ontology_folder_info = info_classes.OntologyFolderInfo(ontology_folder_info)
    save_path = ontology_folder_info.ontology_info(frame='ignore').default_tree_path()
    spec = ontology_folder_info.configuration()
    kwargs = spec['downloading_arguments']
    url = spec['downloading_urls']['ontology']
    url = url.format(**kwargs)
    urllib.request.urlretrieve(url, str(save_path))

    # refactor xml
    old_root = ElementTree.parse(save_path).getroot()
    new_root = ElementTree.Element('blank_node')
    _recursively_refactor_default_ontology_node(new_root, old_root)
    ElementTree.ElementTree(new_root[0]).write(save_path, encoding='utf8')
    print(f"default ontology written to {save_path}")


def _recursively_refactor_default_ontology_node(new_node: ElementTree.Element, old_node: ElementTree.Element,
                                                lvl: int = 0) -> None:
    """
    Makes xml ontology for brain atlas more readable,
    filters out unimportant details (display options, etc).
    Assigns hierarchy levels to each structure.
    Each structure in the initial tree is represented by a single node (unlike the initial tree)
    :param new_node: root node for new tree (function will append graph nodes to it
    :param old_node: old tree root, that stays unchanged
    :param lvl: starting value for tree depth counter
    """
    for old_elem in old_node:
        new_elem = new_node.makeelement('structure', {
            'name': old_elem[4].text.strip('"').title().replace(',', ''),
            'acronym': old_elem[3].text.strip(' "'),
            'id': old_elem[0].text,
            'level': str(lvl),
            'filename': '',
            })
        new_node.append(new_elem)
        try:  # try to get node children and recur
            _recursively_refactor_default_ontology_node(new_elem, old_elem[10], lvl + 1)
        except IndexError:  # means there are no children
            pass


def download_svgs(ontology_folder_info: info_classes.ontology_folder_info_like) -> None:
    """
    Downloads svg atlas masks from Allen Brain Institute.
    Names for files and atlas section ids are specified by
    `downloading_arguments/svg_names_and_slice_ids` entries
    in the configuration of ontology folder.
    >!Note that atlas sections are hard-coded to be matched
     with brain images based on `frame` metadata entry.
     Each atlas section corresponds to one and only `frame` value and has the name `{frame_value}.svg`.
    """
    ontology_folder_info = info_classes.OntologyFolderInfo(ontology_folder_info)
    save_folder = ontology_folder_info.svgs_folder()
    spec = ontology_folder_info.configuration()
    kwargs = spec['downloading_arguments']
    url = spec['downloading_urls']['svg']
    for name, slice_id in kwargs['svg_names_and_slice_ids'].items():
        url_complete = url.format(**kwargs, slice_id=slice_id)
        save_path = (save_folder / name).with_suffix('.svg')
        urllib.request.urlretrieve(url_complete, save_path)
        print(f'downloaded: {save_path}')
