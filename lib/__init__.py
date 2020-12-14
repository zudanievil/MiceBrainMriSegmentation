import pathlib
import yaml

__version__ = '1.0.0'
__all__ = ['load_config']


def load_config(*modules: 'module or module.__dict__', add_to_global: dict = dict(),
                config_path: pathlib.Path = None) -> None:
    """
    loads config to _LOC and _GLOB dicts of the modules, clearing them first
    config_path defaults to loading default_config.yml from the directory with this module
    """
    modules = list(modules)
    try:
        i = 0
        while True:
            if not isinstance(modules[i], dict):
                modules[i] = modules[i].__dict__
            if '_IMPORTS_THAT_NEED_CONFIG' in modules[i]:
                for imp in modules[i]['_IMPORTS_THAT_NEED_CONFIG']:
                    if not isinstance(imp, dict):
                        imp = imp.__dict__
                    if imp not in modules:
                        modules.append(imp)
            i += 1
    except IndexError:
        pass
    config_path = config_path or (pathlib.Path(__file__).parent / 'default_config.yml')
    with config_path.open('rt') as f:
        config = yaml.safe_load(f.read())
    config['global'].update(add_to_global)
    for m in modules:
        m_name = pathlib.Path(m['__file__']).stem
        m['_LOC'].clear()
        m['_LOC'].update(config[m_name])
        m['_GLOB'].clear()
        m['_GLOB'].update(config['global'])

