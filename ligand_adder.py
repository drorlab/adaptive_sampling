#!/usr/bin/env python
"""
Adds ligands to a file
"""
import numpy as np
import os
import sys
import random
import subprocess
from beak.msm import utils
from beak.analyze import get_min_distance
from configparser import ConfigParser
from Dabble import DabbleBuilder, VmdSilencer
from tempfile import TemporaryDirectory
from vmd import atomsel, molecule, vmdnumpy

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               HELPER CLASS                                   #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class LigandAdder(object):
    """
    Adds ligands to a molecule with one ligand set.

    Args:
        inputfile (str): Path to MAE file with one ligand
        configfile (str): Path to config file for this sampling run
    """
    def __init__(self, inputfile, configfile):

        self.molid = molecule.load("mae", inputfile)
        inputname = os.path.split(inputfile)[1]
        self.repindex = int(inputname.split("_")[0])
        self.sampled_clusters = [int(inputname.split("_")[2].replace(".mae",
                                                                      ""))]
        config = ConfigParser(interpolation=None)
        config.read(configfile)
        self.config = config
        self.gen = config.getint("production", "generation")
        self.dabbleopts = config["dabble"]
        self.outdir = os.path.join(os.path.abspath(self.config["system"]["rootdir"]),
                                   "systems",
                                   str(self.gen+1))

        # Set up relevant variables
        self.msm = utils.load(os.path.join(config["system"]["rootdir"],
                                           "production",
                                           str(self.gen),
                                           "mmsm_G%d.pkl" % self.gen))
        self.clusters = utils.load(os.path.join(config["system"]["rootdir"],
                                                "production",
                                                str(self.gen),
                                                "testing.mcluster.pkl"))
        inc_equil = config.getboolean("model", "include_equilibration",
                                      fallback=True)
        self.prodfiles = utils.get_prodfiles(generation=self.gen,
                                             rootdir=config["system"]["rootdir"],
                                             new=False,
                                             equilibration=inc_equil)

        criteria = config["model"].get("criteria", "hub_scores")
        if criteria == "hub_scores":
            self.weights = utils.load(os.path.join(config["system"]["rootdir"],
                                                   "production",
                                                   str(self.gen),
                                                   "mmsm_scores.pkl"))
        elif criteria == "populations":
            self.weights = self.msm.populations_

        elif criteria == "counts":
            def _count_one(self, num):
                return np.sum([len(np.where(c == num)[0]) for c in self.clusters])
            self.weights = sorted(range(self.config.getint("model", "macrostates")),
                                  key=lambda x: _count_one(self, x))

        self.bulk_clusters = np.argsort(self.weights)[-5:]

    #===========================================================================

    def go(self):
        """
        Actually runs
        """
        # Set the user field of actually placed ligand to nonzero
        # this is the prettiest way to do ligand placement, but the user
        # field doesn't save. We temporarily communicate this with zero cooods
        atomsel("all").set("user", 1.0)
        atomsel("same fragment as resname %s and (x=0. and y=0. and z=0.)"
                % self.config["system"]["ligands"].replace(",", " "),
                molid=self.molid).set("user", 0.0)

        # Add ligands and Dabble
        with VmdSilencer(output=os.path.join(self.config["system"]["rootdir"],
                                             "systems",
                                             str(self.gen+1),
                                             "generator.log")):
            self.add_ligands_and_dabble()

        # Start the simulation
        self.start_jobs()

        # Check for output files from other adders. If we're all done,
        # increment generation
        for idx in range(1, self.config.getint("model", "samplers")+1):
            if not os.path.isfile(os.path.join(self.outdir, "%d.prmtop" % idx)):
                return

        self.config["production"]["generation"] = str(self.gen+1)
        with open(os.environ.get("CONFIG"), 'w') as configfile:
            self.config.write(configfile)

    #===========================================================================

    def start_jobs(self):
        """
        Starts all the production jobs
        """
        inpdir = os.path.join(self.config["system"]["rootdir"],
                              "systems", str(self.gen+1))
        command = [
            os.path.join(os.environ["PI_HOME"], "software", "submit_new",
                                     "submit_new"),
            "-p", os.path.join(inpdir, "%d.prmtop" % self.repindex),
            "-c", os.path.join(inpdir, "%d.inpcrd" % self.repindex),
            "-d", self.config["system"]["rootdir"],
            "-s", "80",
            "-t", self.config["production"]["runlength"],
            "-q", self.config["production"]["queue"],
            "-j", "%s-G%d-r%d" % (self.config["system"]["jobname"],
                                  self.gen+1, self.repindex),
            "-r", str(self.gen+1),
            "--%s" % self.config["production"]["pressure"],
            "--msm", str(self.repindex),
            "--gpus", "1",
        ]
        if self.config.getboolean("production", "hmr", fallback=True):
            command.append("--hmr")

        # Could put above with fallback but this is less likely to take down
        # all my jobs lol
        if self.config.get("production", "extra_options", fallback=None):
            command.extend(["--extra-production-options",
                            self.config["production"]["extra_options"]])


        subprocess.check_call(command)

    #==========================================================================

    def _add_one_ligand(self, clust, permissive=False):
        """
        Adds a single ligand to this frame using the greedy criteria:
            - lowest hub score out of candidate hub score array
            - not too close to existing sampled ligand
        For now, try 10 times to find one, or give up and go random

        Args:
            molid (int): VMD molecule ID we're adding ligands too
            clust (int): Cluster to sample
            permissive (bool): Relaxes criteria about ligand being inside
                the box

        Returns:
            (bool) Whether a ligand was successfully added
        """
        # Get a frame representing this cluster
        nligs = self.config.getint("system", "num_ligands")
        fileindex, frameindex, ligidx = utils.get_frame(clust,
                                                        dataset=self.clusters,
                                                        nligs=nligs)
        framid = utils.load_trajectory(filename=self.prodfiles[fileindex],
                                       config=self.config,
                                       frame=frameindex)

        # Get a selection for relevant ligand from this frame
        lignames = self.config["system"]["ligands"].split(',')
        ligids = sorted(set(atomsel("resname %s"
                                    % " ".join(lignames),
                                    molid=framid).get("residue")))
        ligsel = atomsel("same fragment as residue %d"
                         % ligids[ligidx], framid)

        # Check ligand isn't out of the box, which can happen sometimes
        # with solvent ligands. This can result in some weird dabbled
        # systems that later crash
        box = [float(x)/2. for x in self.dabbleopts["dimensions"].split(',')]
        #pylint: disable=too-many-boolean-expressions
        if not permissive and (\
           max(ligsel.get("x")) > box[0] or \
           min(ligsel.get("x")) < -box[0] or \
           max(ligsel.get("y")) > box[1] or \
           min(ligsel.get("y")) < -box[1] or \
           max(ligsel.get("z")) > box[2] or \
           min(ligsel.get("z")) < -box[2]):
            print("   Rejected out of box")
            return False
        #pylint: enable=too-many-boolean-expressions

        # Check ligand distance from protein so no overlap happens
        protsel = atomsel("protein and noh and not same fragment as "
                          "resname %s" % " ".join(lignames), self.molid)
        mind = get_min_distance(ligsel, protsel)
        if mind < float(self.dabbleopts["minprotdist"]):
            print("   Rejected too close to protein: %f" % mind)
            return False

        # Check ligand distance to already chosen ligands in molid
        ligfrags = set(atomsel("resname %s and user 1.0"
                               % " ".join(lignames),
                               molid=self.molid).get("fragment"))
        for frag in ligfrags:
            checksel = atomsel("fragment %s" % frag, self.molid)
            mind = get_min_distance(checksel, ligsel)
            if mind < float(self.dabbleopts["minligdist"]):
                print("   Rejected too close to ligands: %f" % mind)
                return False

        # Pick an unset ligand to move
        selres = min(atomsel("resname %s and not user 1.0"
                             % " ".join(lignames),
                             self.molid).get("residue"))
        oldidxs = atomsel("same fragment as residue %d"
                          % selres, self.molid).get("index")
        newidxs = ligsel.get("index")
        if len(oldidxs) != len(newidxs):
            raise ValueError("AHHH LIGANDS")

        # Get desired ligand coordinates with a view into array and copy
        subsetm = vmdnumpy.timestep(self.molid, 0)[min(oldidxs):max(oldidxs)+1]
        subsetf = vmdnumpy.timestep(framid, 0)[min(newidxs):max(newidxs)+1]
        np.copyto(src=subsetf, dst=subsetm)

        # Set the user field of this ligand to indicate it's selected
        atomsel("same fragment as residue %d"
                % selres, self.molid).set("user", 1.0)

        # Do some cleanup and add this ligand to the sampled list
        molecule.delete(framid)
        self.sampled_clusters.append(clust)

        return True

    #==========================================================================

    def _get_random_cluster(self):
        """
        Returns num random clusters to resample, with probability
        inversely proportional to representation

        Args:
            weights (list of float): Weight to apply to each cluster.
                Will be normalized and turned to probabiilities
            msm (MarkovStateModel): MSM to get cluster labels from

        """
        probs = [1./h for h in self.weights]
        probs = [_/sum(probs) for _ in probs]

        probsel = self.msm.inverse_transform(np.random.choice(range(len(probs)),
                                                              size=1,
                                                              p=probs))
        return probsel[0][0]

#==========================================================================


    def _put_ligand_in_solvent(self):
        """
        If the greedy algorithm fails, find a frame from the solvent
        cluster that's not too far away and just put the ligand there.

        Tries 50 times normally, then relaxes restrictions about where
        the ligand can end up. This is more likely to result in a system
        that crashes but at least it's built in the first place.

        Args:
            molid (int): Molecule ID to add a ligand to
            bulk_clusters (list of int): Clusters to consider adding

        Returns:
            (int): Cluster that was selected
        """

        gotagoodframe = False
        counter = 0
        while not gotagoodframe:
            solclust = int(random.choice(self.bulk_clusters))
            gotagoodframe = self._add_one_ligand(clust=solclust,
                                                 permissive=counter > 50)
            if counter > 50:
                print("Permissively trying %d" % solclust)
                sys.stdout.flush()
            counter += 1

        self.sampled_clusters.append(solclust)
        return solclust

#==========================================================================

    def _greedily_add_ligand(self):
        """
        Adds a buncha ligands to each molid using the "greedy" criteria:
            - low hub score (probabilistically chosen)
            - not too close to an existing sampled ligand
        If we try more than 10 different clusters and can't find one that
        matches, just put the ligand at some location in the solvent, as
        determined by a random frame in the bulk cluster
        """


        done = False
        tried = list(self.sampled_clusters)
        while not done:
            # If we've tried more than 20 different clusters, take
            # a bulk one
            if len(tried) > 20:
                break

            # Get a new unsampled cluster, except allow the bulk
            # cluster to be chosen multiple times
            clust = self._get_random_cluster()

            if clust not in self.bulk_clusters  and clust in tried:
                # TODO: permissiveness allow double picking of clusters?
                # May usually fail though, more expensive?
                continue

            done = self._add_one_ligand(clust, permissive=False)

            if not done:
                tried.append(clust)
            sys.stdout.flush()

        # If greediness failed, just place other ligand from bulk
        if not done:
            print("Going for solvent")
            sys.stdout.flush()
            self._put_ligand_in_solvent()

        sys.stdout.flush()

#==========================================================================

    def add_ligands_and_dabble(self):
        """
        Adds remaining ligands to file.
        Saves all molecules to a temporary file and invokes the Dabble
        Builder API. Names them according to the replicate id.

        Args:
            molid (int): Molecule ID to save
        """

        # Add remaining ligands
        for _ in range(self.config.getint("system", "num_ligands")-1): # First already set
            self._greedily_add_ligand()

        # Print out resampler info, for easy backtracking
        print("SAMPLER %d.psf: Clusters: %s" % (self.repindex,
                                                self.sampled_clusters))
        sys.stdout.flush()

        # Write temporary file with correct ligand positions
        filen = os.path.join(self.outdir, "%d_inp.mae" % self.repindex)
        atomsel("all", self.molid).write('mae', filen)

        with TemporaryDirectory(dir=os.environ.get("SCRATCH"),
                                prefix="dabble") as tdir:
            if len(self.dabbleopts.get("topologies", "").strip()):
                topos = self.dabbleopts["topologies"].split(',')
            else:
                topos = None
            if len(self.dabbleopts.get("parameters", "").strip()):
                params = self.dabbleopts["parameters"].split(',')
            else:
                params = None
            builder = DabbleBuilder(solute_filename=filen,
                                    output_filename=os.path.join(self.outdir,
                                                                 "%d.prmtop" % self.repindex),
                                    user_x=float(self.dabbleopts["dimensions"].split(',')[0]),
                                    user_y=float(self.dabbleopts["dimensions"].split(',')[1]),
                                    user_z=float(self.dabbleopts["dimensions"].split(',')[2]),
                                    membrane_system=self.dabbleopts.get("solvent", "DEFAULT"),
                                    forcefield=self.dabbleopts["forcefield"],
                                    extra_topos=topos,
                                    extra_params=params,
                                    exclude_sel=self.dabbleopts.get("excludesel"),
                                    hmassrepartition=self.config.getboolean("production", "hmr", fallback=True),
                                    overwrite=True,
                                    tmp_dir=tdir
                                   )
            builder.write()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               ACTUAL MAIN AREA                               #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":

    # Load all relevant files according to this generation
    if not os.environ.get("INPUT"):
        raise ValueError("No input file in $INPUT")
    if not os.environ.get("CONFIG"):
        raise ValueError("No config file in $CONFIG")

    adder = LigandAdder(inputfile=os.environ.get("INPUT"),
                        configfile=os.environ.get("CONFIG"))
    adder.go()
