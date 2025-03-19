from collections import defaultdict
from datetime import datetime
from inspect import getfullargspec
from time import time
from typing import Dict, Hashable

from .constants import LogLiterals


def run_method_get_io_dict(method_obj, del_self_arg: bool, *argv, **kwargs) -> Dict:
    """
    Run method method_obj with *argv as arguments.
    Create dictionary of all input/ output and other meta data elements to be eventually logged to file.

    :param method_obj: method reference
    :param del_self_arg: True if we shouldn't include `self` variable in output dictionary
    :param argv: Arguments that needs to be passed to method as *argv
    :param kwargs: Arguments that needs to be passed to method as **kwargs

    :return: Dict that has inputs, outputs and meta data to be logged
    """
    args_to_log = defaultdict(dict)

    start_time = time()
    output = method_obj(*argv, **kwargs)
    execution_time = time() - start_time

    # get name of input parameters of method method_obj
    arg_spec = getfullargspec(method_obj)
    arg_names = arg_spec.args
    argv_list = list(argv)

    # Capture all *argv values
    for arg_name, arg_val in zip(arg_names[:len(argv_list)], argv_list):
        if isinstance(arg_val, Hashable) and not (del_self_arg and arg_name == "self"):
            args_to_log[LogLiterals.INPUTS][arg_name] = str(arg_val)

    # Capture all **kwargs values
    args_to_log[LogLiterals.INPUTS].update(kwargs)

    if arg_spec.defaults:
        default_arg_values = list(arg_spec.defaults)
        # For args that don't have any value, set defaults
        arg_with_no_values_count = len(arg_names) - (len(argv_list) + len(kwargs))
        # Number of arguments for which defaults should be used
        defaults_count = min(arg_with_no_values_count, len(default_arg_values))

        # Arguments for which values are not passed but defaults are specified, use defaults
        for arg_names, arg_val in zip(arg_names[-defaults_count:], default_arg_values[-defaults_count:]):
            if isinstance(arg_val, Hashable):
                args_to_log[LogLiterals.INPUTS][arg_name] = str(arg_val)

    args_to_log[LogLiterals.OUTPUTS] = output
    args_to_log[LogLiterals.META][LogLiterals.EXEC_SEC] = execution_time
    args_to_log[LogLiterals.META][LogLiterals.TIMESTAMP] = datetime.now()

    return args_to_log
