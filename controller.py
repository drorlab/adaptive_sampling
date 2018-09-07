#!/usr/bin/env python
#SBATCH --time=12:00:00
#SBATCH --partition=rondror
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --cpus-per-task=1 --tasks=4 --ntasks-per-socket=2
#SBATCH --mem=32GB

"""
Controller for building MSM for ligand binding
"""
import os
import sys
import time
from glob import glob
import numpy as np
import mdtraj as md

from configparser import ConfigParser
from beak.reimagers import reimage_single_dir
from beak.msm import utils, aggregator
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.decomposition import tICA
from msmbuilder.featurizer import MultiligandContactFeaturizer
from msmbuilder.lumping import PCCAPlus
from msmbuilder.msm import MarkovStateModel
from msmbuilder.tpt import hub_scores
from multiprocessing import Pool

sys.path.append("/scratch/PI/rondror/rbetz/adaptivesampling/")
from inpcrd_generator import InpcrdGenerator

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pylint: disable=too-many-instance-attributes
class MSMController(object):
    """
    Controls adaptive sampling and stuffs
    """

    #==========================================================================

    def __init__(self, configfile, **kwargs):
        """
        Creates an MSMcontroller for a given generation

        Args:
            configfile (str): Configuration file to read generation and file info from
            generation (int): Maximum generation to read. If None,
                will be read from config file
            skip_reimage (bool): If reimaging should be skipped, for example
                if recalculating things
            save_extras (bool): Also dumps clusterers
        """
        self.configfile = configfile
        self.config = ConfigParser(interpolation=None) # because % signs in file
        self.config.read(self.configfile)
        self.featurized = self.config.get("system", "featurized",
                                          fallback="featurized_%d")
        self.dir = os.path.abspath(self.config.get("system", "rootdir",
                                                   fallback=os.getcwd()))
        self.ligands = self.config.get("system", "ligands",
                                       fallback="").split(',')
        self.nreps = self.config.getint("model", "samplers", fallback=0)
        self.name = self.config.get("system", "jobname", fallback="sampler")
        self.num_ligands = self.config.getint("system", "num_ligands")

        self.generation = int(kwargs.get("generation",
                                         self.config.getint("production",
                                                            "generation",
                                                            fallback=0)))
        self.currtime = time.time()
        self.save_extras = bool(kwargs.get("save_extras", False))

        # Reimage to get new production files
        # Only reimage if these files haven't been featurized
        reimglob = glob(os.path.join(self.dir, "production",
                                     str(self.generation), "*",
                                     "Reimaged_strip*.nc"))
        if kwargs.get("skip_reimage", False):
            print("  Skipping reimaging on request...")
        elif len(reimglob) >= self.config.getfloat("production", "runpercent") \
                            * self.nreps:
            print("Skipping reimaging... have %d files already" % len(reimglob))
        else:
            self.reimage()

        print("TIME:\treimage\t%f" % (time.time()-self.currtime))
        sys.stdout.flush()
        self.currtime = time.time()

        inc_equil = self.config.getboolean("model", "include_equilibration",
                                           fallback=False)
        self.prodfiles = utils.get_prodfiles(self.generation, self.dir,
                                             equilibration=inc_equil)
        self.new_prodfiles = utils.get_prodfiles(self.generation, self.dir,
                                                 new=True, equilibration=inc_equil)
        # Sanity check
        if not len(self.new_prodfiles):
            raise ValueError("No production files in generation %d"
                             % self.generation)

    #==========================================================================

    def run(self):
        """
        Actually does the hard work of building the MSM and creating the
        next generation
        """
        # Check if a pickled msm already exists. If so, use it to save time
        if os.path.isfile("mmsm_G%d.pkl" % self.generation) and \
           os.path.isfile("testing.cluster.pkl"):
            print("Loading MSM for generation %d:" % self.generation)
            msm = utils.load("mmsm_G%d.pkl" % self.generation)
            if os.path.isfile("testing.mcluster.pkl"):
                mclusters = utils.load("testing.mcluster.pkl")
            else:
                mclusters = utils.load("testing.cluster.pkl")

        else:
            print("GENERATION %d:" % self.generation)
            # Check if tica exists already
            if os.path.isfile("testing.tica.pkl"):
                print("Loading tics...")
                tics = utils.load("testing.tica.pkl")
            elif os.path.isfile("ticad_%d.h5" % self.generation):
                tics = utils.load_features_h5("ticad_%d.h5" % self.generation)
            else:
                # Featurize new trajectories
                print("  Featurizing...")
                sys.stdout.flush()
                self.currtime = time.time()
                features = self.update_features()
                print("TIME:\tfeaturize\t%f" % (time.time()-self.currtime))
                print("Files: %d Features: %d" % (len(self.new_prodfiles), len(features)))

                # Regenerate tics
                print("  tICing...")
                sys.stdout.flush()
                self.currtime = time.time()
                tics = self.generate_tics(features)
                print("TIME:\tticaing\t%f" % (time.time()-self.currtime))

            print("  Clustering...")
            sys.stdout.flush()
            self.currtime = time.time()
            clusters = self.generate_clusters(tics)
            print("TIME:\tcluster\t%f" % (time.time()-self.currtime))
            utils.dump(clusters, "testing.cluster.pkl")

            print("  MSMing...")
            sys.stdout.flush()
            msm, mclusters = self.generate_msm(clusters)
            utils.dump(mclusters, "testing.mcluster.pkl") # DEBUG

        # Resample, if we haven't reached max generation
        if self.generation < self.config.getint("production", "max_generation",
                                                fallback=1000000):
            print("  Sampling and starting...")
            sys.stdout.flush()
            self.currtime = time.time()
            self.generate_next_inpcrds(msm, mclusters)
            print("TIME:\tinpcrd:\t%f" % (time.time()-self.currtime))
        else:
            self.finish_run()

        # The generation is incremented by the last ligand_adder.

        # Indicate that the model has completed successfully
        # This isn't really necessary but whatever
        self.config["model"]["JobID"] = "0"
        with open(self.configfile, 'w') as configfile:
            self.config.write(configfile)

        # Save representative clusters last
        self.currtime = time.time()
        self.save_cluster_means(mclusters)
        print("TIME:\taggregate:\t%f" % (time.time()-self.currtime))

    #==========================================================================

    def _reimage_one(self, rep):
        # Use prmtop instead of psf
        topo = os.path.join(self.dir, "systems", str(self.generation),
                            "%s.prmtop" % rep)
        if not os.path.isfile(topo):
            topo = os.path.abspath(self.config.get("system", "topologypsf"))

        print("Reimaging...")
        retval = reimage_single_dir(topology=topo,
                                    replicate=rep,
                                    revision=self.generation,
                                    skip=1,
                                    alleq=self.config.getboolean("model",
                                                                 "include_equilibration"),
                                    align=True,
                                    stripmask=":POPC|:TIP3|:SOD|:CLA")
        if retval:
            raise ValueError("Reimaging failed for replicate %s" % rep)

    #==========================================================================

    def reimage(self):
        """
        Reimages a current generation in parallel
        """
        olddir = os.getcwd()
        os.chdir(self.dir)

        proddir = os.path.join(self.dir, "production", str(self.generation))
        reps = [d for d in os.listdir(proddir)
                if os.path.isdir(os.path.join(proddir, d))]

        pool = Pool(int(os.environ.get("SLURM_NTASKS", 4)))
        pool.map(self._reimage_one, reps)
        pool.close()
        pool.join()

        os.chdir(olddir)

    #==========================================================================

    def generate_next_inpcrds(self, msm, clusters):
        """
        Writes the input coordinate files for the next generation.
        Each file is in its own numbered directory and called just "inpcrd"
        """
        # Check if inpcrds are already made
        sysdir = os.path.join(self.dir, "systems", str(self.generation+1))
        if len(glob(os.path.join(sysdir, "*.inpcrd"))) == self.nreps:
            print("   Already have samplers... skipping inpcrd_generation")
            return

        # Make directory to contain topologies and inpcrds
        if not os.path.isdir(os.path.join(self.dir, "systems")):
            os.mkdir(os.path.join(self.dir, "systems"))
        gendir = os.path.join(self.dir, "systems", str(self.generation+1))
        if not os.path.isdir(gendir):
            os.mkdir(gendir)

        scores = hub_scores(msm)
        utils.dump(scores, "mmsm_scores.pkl")
        gen = InpcrdGenerator(prodfiles=self.prodfiles,
                              clusters=clusters,
                              msm=msm,
                              scores=scores,
                              config=self.config,
                              criteria=self.config.get("model", "criteria",
                                                       fallback="hub_scores"))
        gen.run()

    #==========================================================================

    def update_features(self):
        """
        Uses the current trajectories to update the features.

        Returns: featurized all trajectories ready for tica
        """

        # Check feature string has correct format (space for generation)
        if "%d" not in self.featurized:
            print("ERROR: Need format string %d in featurized option")
            quit(1)

        # Featurize this generation
        if not os.path.isfile(self.featurized % self.generation) and \
           not os.path.isfile("%s.pkl" % self.featurized % self.generation) and \
           not os.path.isfile("%s.h5" % self.featurized % self.generation):
            featr = MultiligandContactFeaturizer(ligands=self.ligands,
                                                 scheme="closest-heavy",
                                                 protein=None,
                                                 scaling_function=None,
                                                 log=True)
            feated = []
            for traj in self.new_prodfiles:
                topo = utils.get_topology(traj, self.dir)
                if not os.path.isfile(topo):
                    topo = os.path.abspath(self.config.get("system",
                                                           "topologypsf"))
                featme = md.load(traj, top=topo, stride=1)
                # Hilariously this requires a list to be the right output shape
                feated.extend(featr.transform([featme]))

            # Save this feature set, with backwards compatibility for pickle runs
            if ".pkl" in self.featurized:
                utils.dump(feated, self.featurized % self.generation)
            else:
                utils.save_features_h5(feated,
                                       "%s.h5" % self.featurized % self.generation)
        else:
            print("Already have features for generation %d" % self.generation)
            if os.path.isfile("%s.h5" % self.featurized % self.generation):
                feated = utils.load_features_h5("%s.h5" % self.featurized % self.generation)
            else:
                feated = utils.load("%s.pkl" % self.featurized % self.generation)

        # Check feature file isn't empty. If so, delete it and recurse
        if len(feated) == 0:
            print("Empty features generation %d... Regenerating"
                  % self.generation)
            os.remove("%s.h5" % self.featurized % self.generation)
            feated = self.update_features()

        # We only need to update tica with new features this generation
        return feated

    #==========================================================================

    def generate_tics(self, featurized):
        """
        Now tracks tica object and partially fits on it
        to speed up this step a lot by only adding new data rather than re-fitting
        each time.
        reduced dataset

        Returns: tica'd dataset
        """

        if os.path.isfile(os.path.join(self.dir, "tICA_%d.h5"
                                                  % self.generation)):
            ticr = utils.load_tica_h5(os.path.join(self.dir, "tICA_%d.h5"
                                                   % self.generation))

        elif os.path.isfile(os.path.join(self.dir, "tICA.pkl")): # legacy
            ticr = utils.load(os.path.join(self.dir, "tICA.pkl"))

        else:
            ticr = tICA(n_components=self.config.getint("model", "num_tics"),
                        lag_time=self.config.getint("model", "tica_lag"))

        for newfeat in featurized:
            ticr.partial_fit(newfeat)

        utils.save_tica_h5(ticr, os.path.join(self.dir, "tICA_%d.h5"
                                              % self.generation))

        # Now apply tica to the whole feature set.
        # We need to do this to all featurized data again since the tics
        # have changed since we just updated them with new data
        # Do one at a time to save memory.

        ticad = []
        for gen in range(1, self.generation):
            if os.path.isfile("%s.h5" % self.featurized % gen):
                feated = utils.load_features_h5("%s.h5" % self.featurized % gen)
            else:
                feated = utils.load("%s.pkl" % self.featurized % gen)

            ticad.extend(ticr.transform(feated))

        # Add the features we have in memory now
        ticad.extend(ticr.transform(featurized))
        utils.save_features_h5(ticad, "ticad_%d.h5" % self.generation)

        return ticad

    #==========================================================================

    def generate_clusters(self, ticad):
        """
        Updates the cluster data. Needs to be re-done each iteration as
        cluster from previous trajectories may change as we get more data.

        Returns: clustered dataset
        """
        clustr = MiniBatchKMeans(n_clusters=self.config.getint("model",
                                                               "num_clusters"))
        clustered = clustr.fit_transform(ticad)
        if self.save_extras:
            utils.dump(clustr, "microstater.pkl")
        return clustered

    #==========================================================================

    def generate_msm(self, clustered):
        """
        Generates a MSM from the current cluster data

        Returns: Msm
        """
        # Generate microstate MSM
        self.currtime = time.time()
        msm = MarkovStateModel(lag_time=self.config.getint("model", "msm_lag"),
                               reversible_type="transpose",
                               ergodic_cutoff="off",
                               prior_counts=0.000001)
        msm.fit(clustered)
        print("TIME\tmicromsm:\t%f" % (time.time()-self.currtime))
        utils.dump(msm, "msm_G%d.pkl" % self.generation)

        # Lump into macrostates
        self.currtime = time.time()
        pcca = PCCAPlus.from_msm(msm,
                                 n_macrostates=self.config.getint("model",
                                                                  "macrostates"))
        mclustered = pcca.transform(clustered, mode="fill")
        if any(any(np.isnan(x) for x in m) for m in mclustered): #pylint: disable=no-member
            print("WARNING: Unassignable clusters in PCCA with %d macrostates!"
                  % self.config.getint("model", "macrostates"))
        print("TIME\tpccaplus:\t%f" % (time.time()-self.currtime))
        if self.save_extras:
            utils.dump(pcca, "macrostater.pkl")

        # Generate macrostate MSM
        self.currtime = time.time()
        mmsm = MarkovStateModel(lag_time=self.config.getint("model", "msm_lag"),
                                reversible_type="transpose",
                                ergodic_cutoff="off",
                                prior_counts=0.000001)
        mmsm.fit(mclustered)
        print("TIME\tmacromsm\t%f" % (time.time()-self.currtime))
        utils.dump(mmsm, "mmsm_G%d.pkl" % self.generation)

        return mmsm, mclustered

    #==========================================================================

    def save_cluster_means(self, clusters):
        """
        Generates mean cluster locations and saves them.

        Args:
            clusters (list of ndarray): Clusters to save
        """
        updir = os.path.join(self.dir, "clusters")
        if not os.path.isdir(updir):
            os.mkdir(updir)
        updir = os.path.join(updir, str(self.generation))
        if not os.path.isdir(updir):
            os.mkdir(updir)

        # Check if already computed
        if len(glob(os.path.join(updir, "*.dx"))) == \
               self.config.getint("model", "macrostates"):
            print("Skipping density calculation... already done")
            return

        densitor = aggregator.ParallelClusterDensity(prodfiles=self.prodfiles,
                                                     clusters=clusters,
                                                     config=self.config)
        densitor.save(updir)

    #==========================================================================

    def finish_run(self):
        """
        Does clean up stuff once this run is done, aka we've reached
        max_generations. Primarily, removes the config file from the
        cron checker list of files
        """
        print("  Done with final generation G%d!"
              % self.config.getint("production", "max_generation"))

        cronfile = os.path.join(self.config.get("system", "scriptdir"),
                                "ACTIVE")

        with open(cronfile, 'r') as f:
            actives = f.readlines()
        with open(cronfile, 'w') as f:
            for l in [_ for _ in actives
                      if _.strip() != os.path.abspath(self.configfile)]:
                f.write(l)

    #==========================================================================

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create a new generation
if __name__ == "__main__":
    if os.environ.get("CONFIG") is None:
        raise ValueError("Need to set CONFIG to run a controller")

    controller = MSMController(os.environ.get("CONFIG"))
    controller.run()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
