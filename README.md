# StableRunListReader
## Introduction
`StableRunListReader` enriches an AHCAL run-time JSON file with LHC stable-beam intervals, bad-condition intervals, and final AHCAL good intervals.

The expected input is `ahcal_run_times.json`, as produced by `get_ahcal_run_times.py`.
The output JSON adds the following fields to each run.

- `start_utime`, `stop_utime`: run start/stop as Unix timestamps.
- `stable_list`: intervals from LHC `/LHC/BeamData` where `BeamMode == STABLE BEAMS`.
- `excluded_list`: intervals removed from the AHCAL good interval.
- `good_list`: `stable_list - excluded_list`.
- `stable_time_sec`, `excluded_time_sec`, `good_time_sec`: time summaries in seconds.

# Environment Setup
```bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
setupATLAS -c centos7 -m /eos:/eos
asetup Athena,22.0.49
cd /afs/cern.ch/work/<initial>/<username>/your/StableRunListReader
source setup.sh
```

`daq_conditions.py` reads `faser-secret.json` to access the InfluxDB database.
Put `faser-secret.json` in the current directory before running the DAQ check.
The file is local/private and should not be committed to GitHub.

## Processing Flow
1. Read `ahcal_run_times.json`, usually made by `get_ahcal_run_times.py`.
2. Convert each run start/stop time to Unix timestamps.
3. Read LHC `/LHC/BeamData` and put `BeamMode == STABLE BEAMS` intervals into `stable_list`.
4. Drop runs with an empty `stable_list`.
5. If the run configuration is not allowed, add the whole stable interval to `excluded_list`.
6. Add ATLAS luminosity DB coverage gaps to `excluded_list`.
7. Add AHCAL/DAQ InfluxDB counter missing/gap intervals to `excluded_list`.
8. Compute `good_list = stable_list - excluded_list`.
9. Write the enriched JSON.

## Basic Command
```bash
cd /path/to/your/StableRunListReader

./stable_runlist_reader.py \
  -i ahcal_run_times.json \
  -o ahcal_run_times_with_good_intervals.json \
  --influx-no-verify \
  --lumi-acct-tag OflLumiAcct-Run3-008
```

`--lumi-acct-tag OflLumiAcct-Run3-XXX` is the ATLAS luminosity accounting tag.
`OflLumiAcct-Run3-008` is set as the default in `config.py`.
`OflLumiAcct-Run3-010` was tested, but it did not work 

For more details, see [https://lpc.web.cern.ch/cgi-bin/getMassiAnnotations.py]

## Main Command-Line Options

Input and output:

- `-i, --input PATH`: input run-time JSON. The default is `ahcal_run_times.json`.
- `-o, --output PATH`: output JSON. If omitted, the input file is overwritten.
- `--db PATH`: LHC BeamData COOL sqlite DB. The default is the current FASER DBRelease under `/cvmfs/faser.cern.ch`.
- `-v, --verbose`: print extra diagnostic messages while reading conditions.

Run-configuration filters:

- `--run-config-filter-regex REGEX`: only process runs whose `configuration` matches this regex. The default is empty, which disables this pre-filter.
- `--allowed-config-regex REGEX`: configurations not matching this regex are excluded from `good_list` over the stable-beam interval. The default is `AHCALPhysics`.
- `--physics-config-regex REGEX`: alias for `--allowed-config-regex`.

Luminosity checks:

- `--lumi-acct-tag TAG`: ATLAS luminosity accounting tag used with `/TRIGGER/OFLLUMI/LumiAccounting`. The default is `OflLumiAcct-Run3-008`.
- `--lumi-tag TAG`: ATLAS offline luminosity tag used when checking `OflPrefLumi/LBTIME`. The default is `OflLumi-Run3-008`.
- `--no-lumi`: skip the ATLAS luminosity coverage check.

DAQ/InfluxDB checks:

- `--no-daq`: skip the DAQ counter checks.
- `--secret-file PATH`: JSON file with InfluxDB credentials. The default is `faser-secret.json`.
- `--influx-host HOST`, `--influx-port PORT`, `--influx-user USER`, `--influx-password PASSWORD`, `--influx-database NAME`: override the InfluxDB connection settings.
- `--influx-no-verify`: disable HTTPS certificate verification for InfluxDB.
- `--daq-bin-seconds SECONDS`: InfluxDB query bin size. The default is `1.0`.
- `--daq-max-gap-seconds SECONDS`: DAQ counter gaps larger than this are excluded. The default is `30.0`.
- `--required-measurement NAME`: require this InfluxDB counter measurement during stable beams. This option can be repeated. If it is used, the default `ahcaleventreceiver00-EventNumber` measurement is not added automatically.

## Input JSON
The input is a JSON list. Each entry must contain at least:

```json
{
  "configuration": "AHCALPhysics_NoVeto",
  "run_number": 22127,
  "start_time": "2026-03-07T14:56:08.000000",
  "stop_time": "2026-03-07T21:29:42.000000"
}
```

`run_number`, `start_time`, and `stop_time` are required.
`configuration` is used for run-configuration filtering and exclusions.

## Stable List Condition
`stable_list` is the base interval list used to calculate AHCAL good time.

The condition is:
```text
LHC /LHC/BeamData BeamMode == STABLE BEAMS
```

Only stable-beam intervals overlapping the AHCAL run are kept.
Each interval is clipped to the run start and stop time.
Therefore, a `stable_list` entry never extends outside the run interval.
If a run has no overlapping stable-beam interval, the run is dropped from the output.

## Luminosity Condition
The script checks ATLAS luminosity DB coverage during stable beams.
It does not integrate luminosity or calculate delivered/recorded luminosity values.

If a stable-beam interval has no luminosity record, that part is added to `excluded_list`.
The reason is:
```text
Missing ATLAS luminosity
```

By default, the recommended accounting folder/tag is used:
```bash
--lumi-acct-tag OflLumiAcct-Run3-008
```

Disable this check with:
```bash
--no-lumi
```

## Output JSON
Example output entry:

```json
{
  "configuration": "AHCALPhysics_NoVeto",
  "run_number": 22127,
  "start_time": "2026-03-07T14:56:08.000000",
  "stop_time": "2026-03-07T21:29:42.000000",
  "start_utime": 1772895368.0,
  "stop_utime": 1772918982.0,
  "stable_list": [{"start_utime": 1772895482.0, "stop_utime": 1772918943.0}],
  "excluded_list": [],
  "good_list": [{"start_utime": 1772895482.0, "stop_utime": 1772918943.0}],
  "stable_time_sec": 23461.0,
  "excluded_time_sec": 0.0,
  "good_time_sec": 23461.0
}
```

## DAQ Condition
There are two DAQ filters in `daq_conditions.py`.

1. **Missing DAQ counter**: there is no valid  counter value in the stable interval.
2. **DAQ counter gap**: the DAQ counter has a time gap larger than `--daq-max-gap-seconds`.

By default, only this measurement is required:
```text
ahcaleventreceiver00-EventNumber
```

The exclusion reasons are written as:
```text
Missing DAQ counter <measurement>
DAQ counter gap <measurement>
```

If you want to require another measurement, repeat `--required-measurement`.
```bash
--required-measurement ahcaleventreceiver00-EventNumber \
--required-measurement ahcaleventreceiver00-GoodEventsCount
```

If any `--required-measurement` option is given, the default EventNumber measurement is not added automatically.
List all measurements that should be required.

Disable the DAQ check with:
```bash
--no-daq
```
