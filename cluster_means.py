#!/usr/bin/env python
"""
Generates cluster means and coordinate data for ligands.
Makes it possible to not visualize entire trajectories since
they won't fit in memory any more
"""
import os
from beak.msm import visualizers, sampler
from configparser import ConfigParser
from msmbuilder.utils import load
from vmd import atomsel

#==============================================================================

def save_representative_frames(sample, outdir, msm=None, clusters=None):
    """
    Saves a frame of protein and ligand representing each cluster
    in the sampler in a given directory

    Args:
        sample (Sampler): MSM trajectory object
        outdir (str): Location for saved cluster files
        msm (MarkovStateModel): MSM to use
        clusters (clusters): Cluster stuff
    """
    if msm is None:
        msm = sample.mmsm
    if clusters is None:
        clusters = sample.mclust

    protsel = "(protein or resname ACE NMA) and not same fragment as resname %s" \
              % " ".join(sample.ligands)

    fn = open(os.path.join(outdir, "rmsds"), 'w')

    for cl in msm.mapping_.values():
        print("On cluster: %s" % cl)
        x = visualizers.get_representative_ligand(sample, cl, clusters)
        if x is None: continue
        lg, rms = x
        m, f, l = lg
        print("   Molid: %d\tFrame: %d\tLigand: %d" % (m,f,l))
        fn.write("%s\t%f\n" % (cl, rms))
        atomsel("(%s) or (same fragment as residue %d)" % (protsel, l),
                molid=m, frame=f).write(
                    "mae", os.path.join(outdir, "%s.mae" % cl))

    fn.close()

#==============================================================================

if __name__ == "__main__":

    # Parse arguments
    if not os.environ.get("CONFIG"):
        print("Usage: env CONFIG= config file")
        quit(1)

    cfg = ConfigParser(interpolation=None)
    cfg.read(os.environ.get("CONFIG"))

    # Load optional env variables files
    if os.environ.get("GEN"):
        prevgen = int(os.environ.get("GEN"))
    else:
        prevgen = cfg.getint("production", "generation")-1

    if os.environ.get("STRIDE"):
        stride = int(os.environ.get("STRIDE"))
    else:
        stride = 1

    if os.environ.get("MSM"):
        mmsm = load(os.environ.get("MSM"))
    else:
        mmsm = load(os.path.join(cfg["system"]["rootdir"], "production",
                                 str(prevgen), "mmsm_G%d.pkl" % prevgen))
    if os.environ.get("CLUST"):
        clust = load(os.environ.get("CLUST"))
    else:
        clust = load(os.path.join(cfg["system"]["rootdir"], "production",
                                  str(prevgen), "testing.mcluster.pkl"))
    clust = [c[::stride] for c in clust]

    # Make directory structure
    if os.environ.get("CDIR"):
        outdir = os.environ.get("CDIR")
    else:
        outdir = os.path.join(cfg["system"]["rootdir"], "clusters",
                              str(prevgen))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    updir = os.path.join(cfg["system"]["rootdir"], "clusters")
    if not os.path.isdir(updir):
        os.mkdir(updir)
    updir = os.path.join(updir, str(prevgen))
    if not os.path.isdir(updir):
        os.mkdir(updir)

    # Load data
    # Do one generation before because generation always lags
    samp = sampler.Sampler(configfile=os.environ.get("CONFIG"),
                           generation=prevgen,
                           stride=stride)

    save_representative_frames(samp,
                               outdir,
                               msm=mmsm,
                               clusters=clust)

#==============================================================================

