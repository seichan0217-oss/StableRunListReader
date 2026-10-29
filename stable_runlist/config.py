"""Shared paths and default settings for the analysis.

Defaults are anchored to the checkout, so changing the working directory does
not scatter output files. Installed wheels use the current working directory.
"""

import os
from pathlib import Path

_checkout = Path(__file__).resolve().parent.parent
PROJECT_DIR = _checkout if (_checkout / "pyproject.toml").is_file() else Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
RUNLIST_DIR = OUTPUT_DIR / "runlists"
RUN_TIMES_JSON = RUNLIST_DIR / "ahcal_run_times.json"
RUNLIST_JSON = RUNLIST_DIR / "ahcal_run_times_with_good_intervals.json"
ANALYSIS_DIR = OUTPUT_DIR / "deadtime_trigger_rate"
RAW_RATE_DIR = OUTPUT_DIR / "raw_trigger_rate"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PREPOST_DIR = OUTPUT_DIR / "prepost_veto"
BACKUP_DIR = OUTPUT_DIR / "backups"
STABLE_INTERVALS_CSV = RUNLIST_DIR / "stable_intervals.csv"
GOOD_INTERVALS_CSV = RUNLIST_DIR / "good_intervals.csv"
SELECTION_ANALYSIS_DIR = ANALYSIS_DIR / "selections"
SELECTION_LUMINOSITY_DIR = OUTPUT_DIR / "luminosity"
MODEL_JSON = ANALYSIS_DIR / "summary_anyveto_no_prescale_stat/deadtime_true_rate_model.json"
FAIR_SCRIPTS_DIR = Path(os.environ.get("FAIR_SCRIPTS_DIR", PROJECT_DIR.parent / "FAIR/scripts"))

DEFAULT_INPUT = str(RUN_TIMES_JSON)
DEFAULT_OUTPUT = str(RUNLIST_JSON)
DEFAULT_DB = (
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/sqlite200/ALLP200.db"
)
DEFAULT_INFLUX_HOST = "dbod-faser-influx-prod.cern.ch"
DEFAULT_INFLUX_PORT = 8080
DEFAULT_INFLUX_DATABASE = "$INFLUXDB"
DEFAULT_INFLUX_USER = "$INFLUXUSER"
DEFAULT_INFLUX_PASSWORD = "$INFLUXPW"
DEFAULT_INFLUX_VERIFY_SSL = False
DEFAULT_SECRET_FILE = str(PROJECT_DIR / "faser-secret.json")
DEFAULT_DAQ_BIN_SECONDS = 1.0
DEFAULT_DAQ_MAX_GAP_SECONDS = 10.0
DEFAULT_REQUIRED_MEASUREMENTS = [
    "ahcaleventreceiver00-EventNumber",
]
DEFAULT_LUMI_TAG = "OflLumi-Run3-008"
DEFAULT_LUMI_ACCT_TAG = "OflLumiAcct-Run3-008"
DEFAULT_EFFECTIVE_LUMI_ANY_VETOED_MEASUREMENT = "tlureceiver00-TriggerVetoed"
DEFAULT_EFFECTIVE_LUMI_NON_VETOED_MEASUREMENT = "tlureceiver00-Trigger_i5_0_i6_0"
DEFAULT_EFFECTIVE_LUMI_DEADTIME_MS = 1.22
DEFAULT_EFFECTIVE_LUMI_AUTO_TRIGGERED_MS = 3.9
DEFAULT_PRE_VETO_MEASUREMENT = "tlureceiver00-PreVetoTriggers"
DEFAULT_POST_VETO_MEASUREMENT = "tlureceiver00-PostVetoTriggers"
DEFAULT_VETO_COUNTER_POST_RUN_SECONDS = 10.0
