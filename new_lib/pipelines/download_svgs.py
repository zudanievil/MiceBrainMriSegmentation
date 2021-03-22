import urllib.request
import pandas
import numpy
from xml.etree import ElementTree
from ..core import info_classes


def main(ontology_folder_info: info_classes.ontology_folder_info_like):
    ontology_folder_info = info_classes.OntologyFolderInfo(ontology_folder_info)
    download_slice_ids_table(ontology_folder_info)
    download_default_ontology(ontology_folder_info)
    download_svgs(ontology_folder_info)
   
 
def download_slice_ids_table(ontology_folder_info: info_classes.OntologyFolderInfo):
    save_path = ontology_folder_info.folder() / 'slice_ids.txt'
    # download
    spec = ontology_folder_info.specification()
    kwargs = spec['downloading_arguments']
    url = spec['downloading_urls']['slice_ids']
    url = url.format(**kwargs)
    urllib.request.urlretrieve(url, save_path)
    # extract slice ids
    root = ElementTree.parse(save_path).getroot()
    slice_ids = []
    for node in root.iter(tag='id'):
        slice_ids.append(int(node.text))
    del root
    slice_ids.reverse()
    # zip with a coordinate range
    slice_ids = numpy.array(slice_ids)
    coords = numpy.linspace(**kwargs['slice_coordinates'], num=len(slice_ids))
    t = pandas.DataFrame()
    t['ids'] = slice_ids
    t['coo'] = coords

    t.to_csv(save_path, sep='\t', index=False)


def download_default_ontology(ontology_folder_info: info_classes.OntologyFolderInfo) -> None:
    save_path = ontology_folder_info.ontology_info(frame='ignore').default_tree_path()
    spec = ontology_folder_info.specification()
    kwargs = spec['downloading_arguments']
    url = spec['downloading_urls']['ontology']
    url = url.format(**kwargs)
    urllib.request.urlretrieve(url, str(save_path))

    # refactor xml
    old_root = ElementTree.parse(save_path).getroot()
    new_root = ElementTree.Element('blank_node')
    _recursively_refactor_default_ontology_node(new_root, old_root)
    ElementTree.ElementTree(new_root[0]).write(save_path, encoding='utf8')


def _recursively_refactor_default_ontology_node(new_node: ElementTree.Element, old_node: ElementTree.Element,
                                                lvl: int = 0):
    for old_elem in old_node:
        new_elem = new_node.makeelement('structure', {
            'name': old_elem[4].text.strip('"').title().replace(',', ''),
            'acronym': old_elem[3].text.strip(' "'),
            'id': old_elem[0].text,
            'level': str(lvl),
            'filename': '',
            })
        new_node.append(new_elem)
        try:
            _recursively_refactor_default_ontology_node(new_elem, old_elem[10], lvl + 1)
        except IndexError:
            pass


def download_svgs(ontology_folder_info: info_classes.OntologyFolderInfo) -> None:
    save_folder = ontology_folder_info.svgs_folder()
    spec = ontology_folder_info.specification()
    kwargs = spec['downloading_arguments']
    url = spec['downloading_urls']['svg']
    for name, slice_id in kwargs['svg_names_and_slice_ids'].items():
        url_complete = url.format(**kwargs, slice_id=slice_id)
        save_path = (save_folder / name).with_suffix('.svg')
        urllib.request.urlretrieve(url_complete, save_path)
        print(f'downloaded: {save_path}')
