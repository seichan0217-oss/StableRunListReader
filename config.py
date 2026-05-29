import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parents[1]
COOL_PYTHON_DIR = WORK_DIR / "cool-conditions" / "python"
if str(COOL_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(COOL_PYTHON_DIR))

DEFAULT_INPUT = "ahcal_run_times.json"
DEFAULT_DB = (
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/sqlite200/ALLP200.db"
)
DEFAULT_INFLUX_HOST = "dbod-faser-influx-prod.cern.ch"
DEFAULT_INFLUX_PORT = 8080
DEFAULT_INFLUX_DATABASE = "$INFLUXDB"
DEFAULT_INFLUX_USER = "$INFLUXUSER"
DEFAULT_INFLUX_PASSWORD = "$INFLUXPW"
DEFAULT_SECRET_FILE = "faser-secret.json"
DEFAULT_REQUIRED_MEASUREMENTS = [
    "ahcaleventreceiver00-EventNumber",
]
DEFAULT_LUMI_TAG = "OflLumi-Run3-008"
DEFAULT_LUMI_ACCT_TAG = "OflLumiAcct-Run3-008"
