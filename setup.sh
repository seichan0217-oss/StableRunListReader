#!/usr/bin/bash
# Need an ATLAS release for PyCool/CoolDataReader luminosity checks.
# Example:
#   export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
#   source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
#   setupATLAS -c centos7 -m /eos:/eos
#   asetup Athena,22.0.49
#
# This local setup keeps StableRunListReader runnable from this directory while
# still reusing the CoolDataReader helpers from the neighboring cool-conditions
# checkout.
_stable_runlist_reader_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_stable_runlist_work_dir="$(cd "${_stable_runlist_reader_dir}/.." && pwd -P)"
_cool_conditions_dir="${_stable_runlist_work_dir}/cool-conditions"

export PYTHONPATH="${_stable_runlist_reader_dir}:${_cool_conditions_dir}/python:${PYTHONPATH}"
export PATH="${_stable_runlist_reader_dir}:${PATH}"

unset _stable_runlist_reader_dir
unset _stable_runlist_work_dir
unset _cool_conditions_dir
