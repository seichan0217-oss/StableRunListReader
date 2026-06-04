import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
LOCAL_PYTHON_DIR = PACKAGE_DIR / "python"
if str(LOCAL_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYTHON_DIR))

DEFAULT_INPUT = "ahcal_run_times.json"
DEFAULT_DB = (
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/sqlite200/ALLP200.db"
)
DEFAULT_INFLUX_HOST = "dbod-faser-influx-prod.cern.ch"
DEFAULT_INFLUX_PORT = 8080
DEFAULT_INFLUX_DATABASE = "$INFLUXDB"
DEFAULT_INFLUX_USER = "$INFLUXUSER"
DEFAULT_INFLUX_PASSWORD = "$INFLUXPW"
DEFAULT_INFLUX_VERIFY_SSL = False
DEFAULT_SECRET_FILE = "faser-secret.json"
DEFAULT_DAQ_BIN_SECONDS = 1.0
DEFAULT_DAQ_MAX_GAP_SECONDS = 10.0
DEFAULT_REQUIRED_MEASUREMENTS = [
    "ahcaleventreceiver00-EventNumber",
]
DEFAULT_LUMI_TAG = "OflLumi-Run3-008"
DEFAULT_LUMI_ACCT_TAG = "OflLumiAcct-Run3-008"
