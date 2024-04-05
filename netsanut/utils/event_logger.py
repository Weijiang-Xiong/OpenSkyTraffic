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
    
def setup_logger(name, log_file=None, level=logging.INFO, color=True, stream=True):
    
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

    return logger


if __name__ == "__main__":
    logger = setup_logger("DefaultLogger", "./scratch")
    logger.debug("Debug")
    logger.info("Info")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")