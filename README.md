# StableRunListReader

This project selects stable and good time intervals for AHCAL runs. It also
calculates deadtime, trigger rates, integrated luminosity, and effective luminosity.

## Directory structure

```text
StableRunListReader/
├── stable_runlist/                 # Shared readers, fit functions, and settings
├── scripts/                       # Arguments, analysis steps, and result output
│   ├── extract_ahcal_runs.py
│   ├── make_good_run_list.py
│   ├── analyze_deadtime_rates.py
│   └── calculate_luminosity.py
├── output/
│   ├── runlists/                  # Run information and interval CSV/JSON files
│   ├── deadtime_trigger_rate/     # Fits, trigger counts, and deadtime models
│   └── luminosity/                # Integrated and effective luminosity
├── pyproject.toml                 # Python package settings
├── setup.sh                       # Optional path setup
└── README.md
```

Default paths and settings are in `stable_runlist/config.py`.

## Setup

Use Python 3.9 or later. Step 3 needs PyROOT. Step 2 needs PyCool for luminosity
checks, and step 4 needs it when downloading luminosity-block (LB) data.
At CERN, initialize the ATLAS tools and start a CentOS 7 container for the
Athena release below. The `-m` option makes `/eos` available inside the container.

```bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source "${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh"
setupATLAS -c centos7 -m /eos:/eos
```

Inside the container, set up Athena and go to your checkout. Replace the example
checkout path with your own.

```bash
asetup Athena,22.0.49
cd /eos/path/to/StableRunListReader
python3 -m pip install -e '.[analysis]'
```

If you already have a compatible ATLAS environment, you can skip its setup.

Run the commands below from the project directory.
For InfluxDB access, use a local `faser-secret.json` file or set `INFLUXUSER`,
`INFLUXPW`, and `INFLUXDB`. Do not commit the credential file.
Raw files are read from `/eos/experiment/faser/raw/` by default.

## Run the analysis

### 1. Extract AHCAL runs and stable intervals

```bash
python3 scripts/extract_ahcal_runs.py
```

This reads RunInfo and LHC BeamData. It writes `ahcal_runs.csv`,
`stable_intervals.csv`, and `ahcal_stable_intervals.json` to `output/runlists/`.

### 2. Apply exclusions and make a good list

```bash
python3 scripts/make_good_run_list.py \
  --input output/runlists/stable_intervals.csv
```

This checks DAQ gaps, missing luminosity data, and run configurations.
It writes `good_intervals.csv`, `excluded_intervals.csv`, and
`ahcal_good_intervals.json` to `output/runlists/`.

### 3. Fit deadtime and calculate trigger rates

```bash
python3 scripts/analyze_deadtime_rates.py \
  --input output/runlists/good_intervals.csv \
  --run 22163 \
  --luminosity-csv /path/to/run_22163_timeseries.csv
```

Use a FAIR-format instantaneous luminosity CSV for `--luminosity-csv`.
If this option is omitted, the script looks for an existing CSV in
`../FAIR/scripts/`, then uses `Luminosity.py` there if needed.

Results are saved in `output/deadtime_trigger_rate/selections/<selection_id>/`.
The linear fit of true rate versus luminosity needs at least three valid points
with different luminosities in each configuration group.

### 4. Calculate integrated and effective luminosity

```bash
python3 scripts/calculate_luminosity.py \
  --input output/runlists/good_intervals.csv \
  --run 22163
```

Use the same CSV and run selection as in step 3 to load its results automatically.
Missing LB data are downloaded with PyCool.
Results are saved as `luminosity.csv`, `luminosity.json`, and
`luminosity_total.json` in `output/luminosity/<selection_id>/`.

By default, the JSON contains one summary per run without `luminosity_blocks`.
Luminosity uncertainties are reported only as combined `total_error_minus` and
`total_error_plus` values (including covered-data totals for incomplete runs).
Statistical/systematic components and deadtime-variation values are kept
internally for aggregation but omitted from output. The LB cache remains detailed.

Add `--include-lumiblocks` to include a `luminosity_blocks` list in each run's JSON
record, with the same combined-error format. CSV output remains one row per run.

```bash
python3 scripts/calculate_luminosity.py \
  --input output/runlists/good_intervals.csv \
  --include-lumiblocks
```

The run-list command uses the same default and option. For example, to update
an existing JSON using its stable intervals and cached analysis inputs:

```bash
python3 -m stable_runlist luminosity \
  --run-json output/runlists/goodlist_with_lumi.json
```

Append `--include-lumiblocks` to keep per-LB details, or also use
`--output output/runlists/goodlist_with_lumi_by_lb.json` to save them separately.

For steps 3 and 4, `--input` can be a stable list, a good list, or another interval
CSV in the same format. The start and stop times in each row define the interval
to analyze. `--run` selects a run within that CSV. Omit it to process all runs,
or repeat it to select several runs: `--run 22163 --run 22169`.
For several luminosity CSVs in step 3, use a filename pattern such as
`--luminosity-csv '/path/to/run_{run}_timeseries.csv'`.

Use `--help` with any script to see the other options.
