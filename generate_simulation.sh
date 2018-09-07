#!/bin/bash

if [[ -z "$SHERLOCK" ]]; then
    echo "Error: You're on $(hostname), not sherlock"
    exit 1
fi

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <prmtop> <directory>"
    exit 1
fi

prmtop=$1
dir=$(readlink -e $2)

if [[ ! -f "$1" ]]; then
    echo "No prmtop: $1"
    exit 1
fi

psf=$(readlink -e ${1/prmtop/psf})
sampler="$2/sampler.cfg"
if [[ ! -f "$psf" ]]; then
    echo "No psf: $psf"
    exit 1
fi
if [[ ! -f "$sampler" ]]; then
    echo "No configuration file: $sampler"
    exit 1
fi

# Extract relevant variables from sampler config file
jobname="$(grep "jobname" "$sampler" | awk -F '=' '{print $2}' | tr -d '[:space:]')"
nanoseconds="$(grep "runlength" "$sampler" | awk -F '=' '{print $2}' | tr -d '[:space:]')"
nreps="$(grep "samplers" "$sampler" | awk -F '=' '{print $2}' | tr -d '[:space:]')"
queue="$(grep "queue" "$sampler" | awk -F '=' '{print $2}' | tr -d '[:space:]')"
pressure="$(grep "pressure" "$sampler" | awk -F '=' '{print $2}' | tr -d '[:space:]')"

# HMR is a boolean, default to true
hmr="$(grep "hmr" "$sampler" | awk -F '=' '{print $2}' | tr -d '[:space:]')"
if [[ "$hmr" != "False" ]]; then
    hmrflag="--hmr"
fi

# Print some helpful infos
echo
echo "Using config file:   $sampler"
echo "Job name prefix:     $jobname"
echo "Running in queue:    $queue"
echo "Starting prmtop:     $prmtop"
echo "Pressure control:    $pressure"
echo "HMR increased dt:    $hmr"
echo "Total desired time:  $nanoseconds"
echo "Number of samplers:  $nreps"
echo

mkdir -p "$dir/systems/1"
# Ensure relative symlinks, helps with visualization over sshfs
relpsf="$(realpath --relative-to="$dir/systems/1" "$psf")"

for ((i=1; i<=$nreps; i++)); do
    echo "   Submitting sampler: $i"
    ln -s "$relpsf" "$dir/systems/1/$i.psf"     # Create psf symlink since all psfs are same
    ln -s "${relpsf%.psf}.prmtop" "$dir/systems/1/$i.prmtop"
    ln -s "${relpsf%.psf}.pdb" "$dir/systems/1/$i.pdb"
    ln -s "${relpsf%.psf}.inpcrd" "$dir/systems/1/$i.inpcrd"


    $PI_HOME/software/submit_new/submit_new -r 1 \
                                            --msm $i \
                                            -j "${jobname}-G1-r$i" \
                                            -d "$dir" \
                                            -q "$queue" \
                                            -s 100 \
                                            -t $nanoseconds \
                                            -p "$prmtop" \
                                            -q "$queue" \
                                            --gpus 1 \
                                            "--$pressure" \
                                            $hmrflag
done

# Add this simulation sampler file to the active file
echo "$sampler" >> "$PI_SCRATCH/rbetz/adaptivesampling/ACTIVE"
