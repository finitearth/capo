import logging
from os import makedirs
from os.path import join
from logging.handlers import TimedRotatingFileHandler

from ..constants.str_literals import FileConstants

logging_handlers_list = []


def set_logging_config(log_dirpath: str, mode: str = "offline") -> None:
    """
    This logger should be used when we are running online production scenario

    :param log_dirpath: Path to directory where logg files should be saved.
    :param mode: Specifies whether the mode is `online or `offline`
    :return:
    """
    global logging_handlers_list
    makedirs(log_dirpath, exist_ok=True)
    logging.basicConfig(filename=join(log_dirpath, FileConstants.logfile_name),
                        filemode='a',
                        format=u"%(asctime)s.%(msecs)03d | %(name)-12s | %(funcName)s:\n%(message)s\n",
                        datefmt='%Y-%m-%d,%H:%M:%S',
                        level=logging.NOTSET,
                        force=True,
                        encoding="utf-8")

    if mode == "online":
        daily_split_handler = TimedRotatingFileHandler(FileConstants.logfile_prefix, when="midnight", backupCount=30, encoding="utf-8")
        daily_split_handler.suffix = "%Y%m%d"
        logging_handlers_list = [daily_split_handler]
    else:
        console = logging.StreamHandler()
        console.setLevel(logging.NOTSET)
        logging_handlers_list = [console]


def get_glue_logger(module_name: str) -> logging.Logger:
    """
    Method to get common logger object for module.

    :param module_name: Name of the module.
    :return: Logger object, which can be used for logging
    """
    global logging_handlers_list

    logger = logging.getLogger(module_name)
    for handler in logging_handlers_list:
        logger.addHandler(handler)
    # TODO: Add handler to log to app insights if Azure connection is ON

    return logger

