import logging
import logging.config
import sys
from pathlib import Path
import datetime


def set_logging():
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    logging.config.dictConfig({
        "version": 1,
        "formatters": {
            "named": {
                "format": "%(levelname)-8s %(message)s"
            }
        },
        "handlers": {
            "console-named": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "named"
            },
            "file-named": {
                "class": "logging.FileHandler",
                "filename": results_dir / f"results_{datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}.txt",
                "formatter": "named"
            }
        },
        "loggers": {
            "evaluate_performance_gmap_logger": {
                "level": "DEBUG",
                "handlers": ["console-named", "file-named"]
            }
        },
    })
