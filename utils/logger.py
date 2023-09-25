#############################################################################
#                                 Packages                                  #
#############################################################################

# system
import logging
import os

#############################################################################
#                                  Script                                   #
#############################################################################


def get_logger(name: str):
    """
    Advanced use of Logging

    Args:
        name (str): name we wish to assign to the custom logger. This differentiates log messages from the pipeline.
    """
    
    logger = logging.getLogger(name)

    # Change the format of the displayed message
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)-8s %(message)s")       

    # Add stream 
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler) 

    logger.setLevel(os.environ.get("LOG_LEVEL", "DEBUG").upper())
    logger.propagate = False

    return logger
