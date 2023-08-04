import logging
import coloredlogs
import os 

logger = logging.getLogger(__name__)
# logger.propagate = False

coloredlogs.DEFAULT_FIELD_STYLES = {
    "asctime": {"color": "green"},
    "hostname": {"color": "magenta"},
    "name": {"color": "blue"},
    "programname": {"color": "cyan"},
    "funcName": {"color": "blue"},
}
coloredlogs.install(
    level="DEBUG",
    logger=logger,
    fmt="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    datefmt="%H:%M:%S",
)

def save_log(save_path):
    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(logging.DEBUG)
    # Add the file handler to the logger
    logger.addHandler(file_handler)

# logger.setLevel(logging.DEBUG)

# Create a file handler and set its level to DEBUG
# file_handler = logging.FileHandler("my_log_file.log")
# file_handler.setLevel(logging.DEBUG)

# Create a formatter and attach it to the file handler
# formatter = logging.Formatter(
#     "%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
#     datefmt="%H:%M:%S"
# )
# file_handler.setFormatter(formatter)

# Add the file handler to the logger
# logger.addHandler(file_handler)
