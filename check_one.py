#!/usr/bin/env python
import os
import sys

scriptdir = os.path.join(os.environ.get("PI_SCRATCH", "/scratch/PI/rondror"),
                         "rbetz", "adaptivesampling")
sys.path.append(scriptdir)

from checker import AdaptiveSamplingChecker
if len(sys.argv) < 2:
    print("Usage: check_one.py <path to directory with sampler>")

samp = sys.argv[1]
if "sampler.cfg" in samp and not os.path.isfile(samp):
    print("Not a file: %s" % samp)
    quit(1)
elif os.path.isfile(os.path.join(samp, "sampler.cfg")):
    samp = os.path.join(samp, "sampler.cfg")
else:
    print("No sampler.cfg in directory: %s" % samp)
    quit(1)

checker = AdaptiveSamplingChecker(os.path.abspath(samp))
checker.check()

print("Done!")
