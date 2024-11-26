import os
import os.path as osp
import sys
import re
import importlib
import importlib.util

def _import_module(module_path_or_name):
    """Dynamically imports a module from a filepath or a module name."""
    module, name = None, None

    if module_path_or_name.endswith('.py'):

        if not os.path.exists(module_path_or_name):
            raise RuntimeError('File {} does not exist.'.format(module_path_or_name))

        file_name = module_path_or_name
        module_name = os.path.basename(os.path.splitext(module_path_or_name)[0])

        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            # Use importlib to load the module from the file
            spec = importlib.util.spec_from_file_location(module_name, file_name)
            if spec is None:
                raise ImportError(f"Cannot load specification for '{module_name}' from '{file_name}'")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    else:
        module = importlib.import_module(module_path_or_name)
        module_name = module.__name__

    if module:
        name = module_name.split('.')[-1].split('/')[-1]

    return module, name

def load(conf_path, *args, **kwargs):
    """Loads a config."""

    module, name = _import_module(conf_path)
    try:
        load_func = module.load
    except AttributeError:
        raise ValueError("The config file should specify 'load' function but no such function was "
                           "found in {}".format(module.__file__))

    print("Loading '{}' from {}".format(module.__name__, module.__file__))
    # parse_flags()
    return load_func(*args, **kwargs)