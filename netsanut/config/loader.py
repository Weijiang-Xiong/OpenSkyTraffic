""" basically copied form 
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/lazy.py
    it's a masterpiece! 
    
"""

import os 
import uuid
import logging
import importlib
import builtins 
from contextlib import contextmanager
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import List

logger = logging.getLogger("default")

_CFG_PACKAGE_NAME = "netsanut._cfg_loader"

def _random_package_name(filename):
    # generate a random package name when loading config files
    return _CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)

def _cast_to_config(obj):
    # if given a dict, return DictConfig instead
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj

@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
    e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through PathManager, so config files can be in the cloud
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        # NOTE: "from . import x" is not handled. Because then it's unclear
        # if such import should produce `x` as a python module or DictConfig.
        # This can be discussed further if needed.
        relative_import_err = """
Relative import of directories is not allowed within config files.
Within a config file, relative import can only import other config files.
""".replace(
            "\n", " "
        )
        if not len(relative_import_path):
            raise ImportError(relative_import_err)

        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not os.path.isfile(cur_file):
            cur_file_no_suffix = cur_file[: -len(".py")]
            if os.path.isdir(cur_file_no_suffix):
                raise ImportError(f"Cannot import from {cur_file_no_suffix}." + relative_import_err)
            else:
                raise ImportError(
                    f"Cannot import name {relative_import_path} from "
                    f"{original_file}: {cur_file} does not exist."
                )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            spec = importlib.machinery.ModuleSpec(
                _random_package_name(cur_file), None, origin=cur_file
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = _cast_to_config(module.__dict__[name])
                module.__dict__[name] = val
            return module
        
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import

class ConfigLoader:

    @staticmethod
    def load_from_file(filename):
        
        with _patch_import():
            # Record the filename
            module_namespace = {
                "__file__": filename,
                "__package__": _random_package_name(filename),
            }
            with open(filename) as f:
                content = f.read()
            # Compile first with filename to:
            # 1. make filename appears in stacktrace
            # 2. make load_rel able to find its parent's (possibly remote) location
            exec(compile(content, filename, "exec"), module_namespace)

        ret = module_namespace
        # when not specified, only load those that are config objects
        ret = DictConfig(
            {
                name: _cast_to_config(value)
                for name, value in ret.items()
                if isinstance(value, (DictConfig, ListConfig, dict))
                and not name.startswith("_")
            },
            flags={"allow_objects": True},
        )
        
        logger.info("Config loaded from {}".format(filename))
        
        return ret
    
    @staticmethod
    def apply_overrides(cfg, overrides: List[str]):
        
        def safe_update(cfg, key, value):
            parts = key.split(".")
            for idx in range(1, len(parts)):
                prefix = ".".join(parts[:idx])
                v = OmegaConf.select(cfg, prefix, default=None)
                if v is None:
                    break
                if not OmegaConf.is_config(v):
                    raise KeyError(
                        f"Trying to update key {key}, but {prefix} "
                        f"is not a config, but has type {type(v)}."
                    )
            OmegaConf.update(cfg, key, value, merge=True)
            
        # TODO https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/lazy.py#L346
        # use hydra to parse the overrides
        for o in overrides:
            key, value = o.split("=")
            try:
                value = eval(value, {})
            except NameError:
                pass
            safe_update(cfg, key, value)
        
        return cfg 
        
    
    @staticmethod
    def save_cfg(cfg, filename:str = None):
        """ convert the cfg to a python dict and format it
        """
        if not filename.endswith(".py"):
            filename += ".py"
            logger.info("Only py files are supported for config, renaming the file to .py")
        
        def dict_to_str(dictionary, key_indent=1):
            
            """ 
                format a dictionary config to the following 
                
                train = {
                    'checkpoint': '',
                    'device': 'cuda',
                    'output_dir': 'scratch/test'
                    'data': {
                        'adj_type': 'doubletransition',
                        'batch_size': 32,
                        'dataset': 'metr-la'
                    } 
                } 

                data = {
                    'adj_type': 'doubletransition',
                    'batch_size': 32,
                    'dataset': 'metr-la'
                } 

            """
            out_string = ""
            out_string += "{\n"
            
            for key, value in dictionary.items():
                out_string += "".join(["\t" for _ in range(key_indent)])
                
                if isinstance(value, dict):
                    # don't call repr for the second argument because the braces { } and indents needs to be 
                    # recognized as the codes of a dictionary value, not string like '{' '}' '\t' '\n'
                    out_string += "{}: {},\n".format(repr(key), dict_to_str(value, key_indent=key_indent+1))
                else:
                    # use repr to keep the quotation marks of the returned string
                    out_string += "{}: {},\n".format(repr(key), repr(value))
                    
            out_string += "".join(["\t" for _ in range(key_indent-1)])
            out_string += "}"
            
            return out_string

        with open(filename, "w") as f:
            cfg_dict = OmegaConf.to_container(cfg)
            for key, value in cfg_dict.items():
                f.write("{} = {} \n\n".format(key, dict_to_str(value)))
        
        logger.info("Config saved to {}".format(filename))