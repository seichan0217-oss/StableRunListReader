#!/usr/bin/bash
# Need an ATLAS release for PyCool luminosity checks.
# Example:
#   export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
#   source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
#   setupATLAS -c centos7 -m /eos:/eos
#   asetup Athena,22.0.49
#
# This local setup keeps StableRunListReader runnable from this directory. The
# CoolDataReader helpers it needs are vendored under StableRunListReader/python.
_stable_runlist_reader_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export PYTHONPATH="${_stable_runlist_reader_dir}:${_stable_runlist_reader_dir}/python:${PYTHONPATH}"
export PATH="${_stable_runlist_reader_dir}:${PATH}"

unset _stable_runlist_reader_dir
