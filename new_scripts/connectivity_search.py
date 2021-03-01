import urllib.request
import pandas as pd
import yaml



url = 'http://connectivity.brain-map.org/projection/csv?criteria=service::mouse_connectivity_injection_structure' \
      '[injection_structures$eq8,304325711][primary_structure_only$eqtrue]'
name_col, connect_col = 'structure-name', 'injection-structures'


def allen_atlas_connectivity_csv_to_adjacency_matrix(load_path, save_path):
    assert load_path != save_path
    t = pd.read_csv(load_path, sep=',')
    idx = pd.Index(t[name_col].map(lambda x: x.title()).unique())
    idx = idx.sort_values()
    t = t[[name_col, connect_col]]
    connectivity = pd.DataFrame(index=idx, columns=idx)  # graph adjacency matrix, src is row, dst is col
    for i in range(len(t)):
        e = 'd=' + str(t.loc[(i, connect_col)]).replace('=>', ':')
        exec(e)  # adds list of dicts named 'd' to locals
        connection_src = t.loc[(i, name_col)].title()
        for connection in locals()['d']:
            connection_dst = connection['name'].title()
            connectivity.loc[(connection_src, connection_dst)] = 1
    connectivity.fillna(value=0, inplace=True)
    connectivity.to_csv(save_path, sep='\t',)


def allen_atlas_connectivity_csv_to_hashmap(load_path, save_path):
    assert load_path != save_path
    t = pd.read_csv(load_path, sep=',')
    idx = t[name_col].map(lambda x: x.title()).unique()
    connectivity = dict.fromkeys(idx)
    for i in range(len(t)):
        e = 'd=' + str(t.loc[(i, connect_col)]).replace('=>', ':')
        exec(e)
        connection_src = t.loc[(i, name_col)].title()
        for connection in locals()['d']:
            connection_dst = connection['name'].title()
            if not connectivity[connection_src]:
                connectivity[connection_src] = set()  # because there are repetitions
            connectivity[connection_src].add(connection_dst)
    connectivity = {k: list(v) for k, v in connectivity.items()}  # because lists are default yaml object
    with open(save_path, 'wt') as f:
        yaml.safe_dump(connectivity, f)


def trace_downstream_connections(hashmap_path, printout_path, structure_name, n_iter):
    with open(hashmap_path, 'rt') as f:
        connectivity = yaml.safe_load(f.read())
    all_sources = {structure_name}
    src = list()
    src.append(dict.fromkeys(all_sources))
    for i in range(n_iter):
        next_src = set()
        for structure_name in src[i]:
            if structure_name in connectivity:
                v = set(connectivity[structure_name])
                try:
                    v.remove(structure_name)
                except ValueError:
                    pass
                next_src.update(v)
                src[i][structure_name] = list(v)
        next_src.difference_update(all_sources)
        all_sources.update(next_src)
        src.append(dict.fromkeys(next_src))
    src.pop(-1)
    with open(printout_path, 'wt') as f:
        yaml.safe_dump(src, f)


if __name__ == '__main__':
    csv_path = 'c:/users/user/downloads/mouse_connectivity.txt'
    csv_save_path = 'c:/users/user/downloads/mouse_adj_matrix.txt'
    hashmap_path = 'c:/users/user/downloads/mouse_adj_tree.yml'
    trace_path = 'c:/users/user/downloads/MOB_connections_trace.yml'
    urllib.request.urlretrieve(url, csv_path)
    allen_atlas_connectivity_csv_to_adjacency_matrix(csv_path, csv_save_path)
    allen_atlas_connectivity_csv_to_hashmap(csv_path, hashmap_path)
    trace_downstream_connections(hashmap_path, trace_path, 'Main Olfactory Bulb', n_iter=3)




