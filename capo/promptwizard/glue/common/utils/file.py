import json
from os.path import join
from typing import Dict, List
import yaml

from ..exceptions import GlueValidaionException


def yaml_to_dict(file_path: str) -> Dict:
    with open(file_path) as yaml_file:
        yaml_string = yaml_file.read()

        try:
            # convert yaml string to dict
            parsed_dict = yaml.safe_load(yaml_string)
        except yaml.scanner.ScannerError as e:
            raise GlueValidaionException(f"There could be some syntax error in yaml written in {file_path}", e)

    return parsed_dict


def yaml_to_class(yaml_file_path: str, cls: type, default_yaml_file_path: str = None):
    """
    Read yaml file present at path `yaml_file_path`, convert it to dictionary using pyyaml's standard methods.
    Then convert this dictionary to class object of class given as `cls`. Further check if user has provided all
    the required fields in `yaml_file_path`. Fields that are missing in `yaml_file_path`, set them with defaults.

    :param yaml_file_path: str
    :param cls: type
    :param default_yaml_file_path: str
    :return:
    """
    if not yaml_file_path:
        yaml_file_path = default_yaml_file_path
    custom_args = yaml_to_dict(yaml_file_path)

    if default_yaml_file_path:
        # If user has not provided all the required arguments, fill them with defaults
        default_args = yaml_to_dict(default_yaml_file_path)
        missing_args = set(default_args) - set(custom_args)
        for key in list(missing_args):
            custom_args[key] = default_args[key]

    try:
        yaml_as_class = cls(**custom_args)
    except TypeError as e:
        raise GlueValidaionException(f"Exception while converting yaml file at {yaml_file_path} "
                                     f"to class {cls.__name__}: ", e)

    return yaml_as_class


def read_jsonl(file_path: str) -> List:
    """
    This function should be used when size of jsonl file is not too big.

    :param file_path:
    :return: All json strings in .jsonl file as a list
    """
    jsonl_list = []
    with open(file_path, "r") as fileobj:
        while True:
            single_row = fileobj.readline()
            if not single_row:
                break

            json_object = json.loads(single_row.strip())
            jsonl_list.append(json_object)
    return jsonl_list


def read_jsonl_row(file_path: str):
    """

    :param file_path:
    :return: Single line from the file. One at a time.
    """
    with open(file_path, "r") as fileobj:
        while True:
            try:
                single_row = fileobj.readline()
                if not single_row:
                    break

                json_object = json.loads(single_row.strip())
                yield json_object
            except json.JSONDecodeError as e:
                print(f"Error while reading jsonl file at {file_path}. Error: {e}")
                continue


def append_as_jsonl(file_path: str, args_to_log: Dict):
    """

    :param file_path:
    :param args_to_log:
    :return:
    """
    json_str = json.dumps(args_to_log, default=str)
    with open(file_path, "a") as fileobj:
        fileobj.write(json_str+"\n")


def save_jsonlist(file_path: str, json_list: List, mode: str = "a"):
    """
    :param json_list: List of json objects
    :param file_path: File location to which we shall save content of json_list list, in jsonl format.
    :param mode: Write mode
    :return: None
    """
    with open(file_path, mode) as file_obj:
        for json_obj in json_list:
            json_str = json.dumps(json_obj, default=str)
            file_obj.write(json_str+"\n")


def str_list_to_dir_path(str_list: List[str]) -> str:
    """
    Return a string which is directory path formed out of concatenating given strings in list `str_list`

    e.g.
    str_list=["dir_1", "sub_dir_1"]
    return "dir_1\sub_dir_1"
    """
    if not str_list:
        return ""

    path = ""
    for dir_name in str_list:
        path = join(path, dir_name)
    return path
