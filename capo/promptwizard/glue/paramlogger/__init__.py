__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from collections import defaultdict
from datetime import datetime
from os import makedirs
from os.path import basename, join
from uuid import uuid4

from . import file_utils as futil
from .constants import LogLiterals
from .utils import run_method_get_io_dict


class ParamLogger:
    def __init__(self, base_path: str = ""):
        """
        :param base_path: Path where all log files would be saved
        """
        self.BASE_PATH = base_path
        if base_path:
            makedirs(self.BASE_PATH, exist_ok=True)

        # Unique `id` for a sample in dataset
        self.SAMPLE_UNQ_ID = None

        # This dictionary can be used, when we want to log output and input of multiple components as a single jsonl
        self.CHAINED_LOG = []

        # When using ParamLogger decorator over a method in a class, should we avoid logging arguement with name `self`
        self.DEL_SELF_ARG = True

    def reset_eval_glue(self, base_path):
        # Path where all log files would be saved
        self.BASE_PATH = base_path
        makedirs(self.BASE_PATH, exist_ok=True)

        # Unique `id` for a sample in dataset
        self.SAMPLE_UNQ_ID = None

        # This dictionary can be used, when we want to log output and input of multiple components as a single jsonl
        self.CHAINED_LOG = []

    def clear_chained_log(self):
        """
        Deletes all previously saved data. Re-initialize CHAINED_LOG with new meta data.
        """
        self.CHAINED_LOG = []

    def dump_chained_log_to_file(self, file_name="chained_logs"):
        """
        Append to file all data collected in CHAINED_LOG as json line.
        Resets CHAINED_LOG to new instance
        """

        file_path = join(self.BASE_PATH, file_name + ".jsonl")
        futil.save_jsonlist(file_path=file_path, json_list=self.CHAINED_LOG)
        self.clear_chained_log()

    def append_dict_to_chained_logs(self, args_to_log):
        self.CHAINED_LOG.append(args_to_log)

    def append_to_chained_log(self, method_obj):
        """
        Execute the method referenced by method_obj. After executing, append the jsonl form of inputs and outputs of
        that method to self.CHAINED_LOG list.

        :param method_obj:
        :return: None
        """
        def wrap(*argv, **kwargs):
            args_to_log = run_method_get_io_dict(method_obj, self.DEL_SELF_ARG, *argv, **kwargs)
            args_to_log[LogLiterals.META][LogLiterals.METHOD_NAME] = method_obj.__name__
            self.CHAINED_LOG.append(args_to_log)
            return args_to_log[LogLiterals.OUTPUTS]
        return wrap

    def log_io_params(self, method_obj, file_name="io_logs"):
        """
        Execute the method referenced by method_obj. After executing, log the inputs and outputs of that method to
        log file.

        :param method_obj: Method reference, that can be executed
        :param file_name: Name of file in which we shall be logging the input output params of method
        :return: None
        """
        def wrap(*argv, **kwargs):
            args_to_log = run_method_get_io_dict(method_obj, self.DEL_SELF_ARG, *argv, **kwargs)
            if not self.SAMPLE_UNQ_ID:
                self.SAMPLE_UNQ_ID = uuid4()
            args_to_log[LogLiterals.ID] = self.SAMPLE_UNQ_ID
            args_to_log[LogLiterals.META][LogLiterals.METHOD_NAME] = method_obj.__name__
            file_path = join(self.BASE_PATH, file_name + ".jsonl")
            futil.append_as_jsonl(file_path=file_path, args_to_log=args_to_log)
            self.SAMPLE_UNQ_ID = None
            return args_to_log[LogLiterals.OUTPUTS]
        return wrap

    def log_io_params_for_method(self, method_obj):
        """
        Execute the method referenced by method_obj. After executing, log the inputs and outputs of that method to
        log file. Name of log file would be the method name

        :param method_obj: Method reference, that can be executed
        :return: None
        """
        def wrap(*argv, **kwargs):
            args_to_log = run_method_get_io_dict(method_obj, self.DEL_SELF_ARG, *argv, **kwargs)
            if not self.SAMPLE_UNQ_ID:
                self.SAMPLE_UNQ_ID = uuid4()
            args_to_log[LogLiterals.ID] = self.SAMPLE_UNQ_ID
            file_path = join(self.BASE_PATH, method_obj.__name__+".jsonl")
            futil.append_as_jsonl(file_path=file_path, args_to_log=args_to_log)
            self.SAMPLE_UNQ_ID = None
            return args_to_log[LogLiterals.OUTPUTS]
        return wrap

    def run_over_logs(self, method_obj):
        """
        Run the method referenced by method_obj over each entry in jsonl file present at location `file_path`.
        `id`, `inputs`, `outputs` fields in jsonl file at `file_path` can be accessed via dummy_id, dummy_input,
        dummy_output parameters respectively.

        :param method_obj:
        :return: None
        """
        def wrap(file_path, dummy_id, dummy_input, dummy_output, dummy_meta, **kwargs):
            eval_file_path = join(self.BASE_PATH, method_obj.__name__ + "_" + basename(file_path))
            args_to_log = defaultdict(dict)

            for json_obj in futil.read_jsonl_row(file_path):
                eval_result = method_obj(None,
                                         json_obj[LogLiterals.ID],
                                         json_obj[LogLiterals.INPUTS],
                                         json_obj[LogLiterals.OUTPUTS],
                                         json_obj[LogLiterals.META],
                                         **kwargs)
                args_to_log[LogLiterals.ID] = json_obj[LogLiterals.ID]
                args_to_log[LogLiterals.EVAL_RESULT] = eval_result
                args_to_log[LogLiterals.META][LogLiterals.TIMESTAMP] = datetime.now()
                futil.append_as_jsonl(file_path=eval_file_path, args_to_log=args_to_log)
        return wrap
