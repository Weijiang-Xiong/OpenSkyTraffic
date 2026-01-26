import logging

from termcolor import colored

class _ColorfulFormatter(logging.Formatter):
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/logger.py
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

def setup_logger(name, log_file=None, level=logging.INFO, color=True, stream=True, propagate=False):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # file handler
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        plainer_formatter = logging.Formatter(
            "[%(asctime)s %(name)s]: %(message)s", datefmt="%m/%d %H:%M:%S"
        )
        fh.setFormatter(plainer_formatter)
        logger.addHandler(fh)
        
    if stream:
        # console handler or stream handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(name),
            )
        else:
            formatter = plainer_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = propagate

    return logger


if __name__ == "__main__":
    logger = setup_logger("DefaultLogger", "./scratch/test_log.log", color=True)
    logger.debug("Debug")
    logger.info("Info")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")

    # we can setup the parent logger and then use child loggers in modules
    parent = setup_logger(name="skytraffic", log_file="./scratch/test_log.log", propagate=False)

    child = logging.getLogger("skytraffic.module")  # no setup needed
    child.info("hello from module")  # goes to skytraffic's file handler via propagation