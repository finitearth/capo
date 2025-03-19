from importlib import import_module
from importlib.metadata import distribution, PackageNotFoundError
import os
from importlib.util import module_from_spec, spec_from_file_location

from os.path import basename, splitext
import subprocess
import sys

from ..constants.log_strings import CommonLogsStr
from ..exceptions import GlueValidaionException
from ..utils.logging import get_glue_logger

logger = get_glue_logger(__name__)


def install_lib_if_missing(lib_name, find_links = None) -> bool:
    """
    Check if library with name `lib_name` is installed in environment. If not, install it in runtime.

    :param lib_name: Name of library
    :return: True if library was installed. False if it was not initially installed and was installed now.
    """
    try:
        version = None
        if "==" in lib_name:
            lib_name, version = lib_name.split("==")
        distri_obj = distribution(lib_name)
        # if version and distri_obj.version != version:
        #     raise GlueValidaionException(f"{lib_name} with version={distri_obj.version} is found. "
        #                                  f"But version needed is {version}", None)
        return True
    except (PackageNotFoundError, GlueValidaionException):
        logger.info(CommonLogsStr.INSTALL_MISSING_LIB.format(lib_name=lib_name))
        with open(os.devnull, 'w') as devnull:
            if find_links:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name, "-f", find_links], stdout=devnull, stderr=devnull)
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name], stdout=devnull, stderr=devnull)

    return False


def str_to_class(class_name: str, import_path: str = None, file_path: str = None):
    """
    For a given `class_name` in string format, return the class instance (not object).
    You need to specify any one of the 2: import_path or file_path. When both are specified `import_path` takes
    precedence.

    :param class_name: Class name, specified as string e.g. CSVReader
    :param import_path: Import path for the specified class_name e.g. llama_index.readers.file
    :param file_path: Path to the file where this class is present. e.g. C:\\dir1\\sub_dir1\\filename.py
    :return: Class
    """

    if import_path:
        cls = getattr(import_module(import_path), class_name)
    elif file_path:
        file_name_without_extsn = splitext(basename(file_path))[0]
        spec = spec_from_file_location(file_name_without_extsn, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
    else:
        cls = getattr(sys.modules[__name__], class_name)

    return cls
