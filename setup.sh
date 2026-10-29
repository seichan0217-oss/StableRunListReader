#!/usr/bin/bash
# Need an ATLAS release for PyCool luminosity checks.
# Example:
#   export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
#   source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
#   setupATLAS -c centos7 -m /eos:/eos
#   asetup Athena,22.0.49
#
# Use the checkout without installing it. COOL helpers are part of the package.
_stable_runlist_reader_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export PYTHONPATH="${_stable_runlist_reader_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${_stable_runlist_reader_dir}/scripts:${_stable_runlist_reader_dir}/build/bin:${PATH}"

unset _stable_runlist_reader_dir
