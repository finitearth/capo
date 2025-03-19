import json
from os.path import join
from typing import Dict, List


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
            single_row = fileobj.readline()
            if not single_row:
                break

            json_object = json.loads(single_row.strip())
            yield json_object


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
