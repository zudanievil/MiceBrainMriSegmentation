import pathlib
from xml.etree import ElementTree
from ..core import info_classes


def list_substructures(ontology_folder_info: info_classes.OntologyFolderInfo,
                        structure_names: 'list[str]') -> 'list[ElementTree.Element]':
    root = ontology_folder_info.ontology_info('').default_tree().getroot()
    result = []
    for name in structure_names:
        found = False
        for node in root.iter('structure'):
            if node.attrib['name'] == name:
                found = True
                break
        if found:
            for subnode in node.iter('structure'):
                result.append(subnode)
    return result


def write_node_list(node_list: 'list[ElementTree.Element]', path: pathlib.Path, indentation: str=' '):
    with path.open('wt') as f:
        for node in node_list:
            s = indentation*node.attrib['level'] + node.attrib['name']
            print(s, file=f)


def main(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like,
         parent_structures_names: 'list[str]' = ['Root']):
    rf = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    ontology_folder_info = rf.ontology_folder_info()
    node_list = list_substructures(ontology_folder_info, parent_structures_names)
    path = rf.structure_list_path()
    write_node_list(node_list, path)
