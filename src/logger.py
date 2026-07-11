import logging
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Create logs directory
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = LOG_DIR / "running.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s')

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

fileHandler = logging.FileHandler(
    filename= LOG_FILE,
    mode='a'
)
fileHandler.setLevel(logging.ERROR)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
