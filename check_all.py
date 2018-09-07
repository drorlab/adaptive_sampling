#!/usr/bin/env python

#SBATCH --job-name=cron_checker
#SBATCH --time=01:00:00
#SBATCH --tasks=1 --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --partition=rondror
#SBATCH --begin=now+30minutes
#SBATCH --mail-type=FAIL --mail-user=rbetz@stanford.edu
#SBATCH --dependency=singleton
#SBATCH --open-mode=append

import os
import sys
import subprocess

scriptdir = os.path.join(os.environ.get("PI_SCRATCH", "/scratch/PI/rondror"),
                         "rbetz", "adaptivesampling")
sys.path.append(scriptdir)

command = [
           "sbatch",
           "--output=checker.log"
          ]
command.append(sys.argv[0])

# Resubmit for 1 hour from now... hack since I can't cron
# Do this here so if it crashes during checking it still runs more
subprocess.call(command)

from checker import AdaptiveSamplingChecker

actives = os.path.join(scriptdir, "ACTIVE")
with open(actives, 'r') as fn:
    samplers = [line.strip() for line in fn.readlines()
                if line.strip() and line.strip()[0] != "#"]

for samp in samplers:
    try:
        checker = AdaptiveSamplingChecker(os.path.abspath(samp))
        checker.check()
    except FileNotFoundError as e:
        print("No config file found. Skipping %s" % samp)

print("Done!")
