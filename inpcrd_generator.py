"""
Generates inpcrds using a greedy algorithm
"""
import os
import numpy as np
import subprocess
from beak.msm import utils
from Dabble import VmdSilencer
from vmd import atomsel, molecule

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               SAMPLER CLASS                                  #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
class InpcrdGenerator(object):
    """
    A generator for inpcrds, uses a greedy algorithm to best select
    ligands in combination that represents as many clusters as possible.

    What this actually does is create a bunch of inpcrds based on the top
    20 "guaranteed" resamplers where the protein and ligand come together.
    The other ligand positions are set to all 0. That structure with only
    protein and ligand(s) is written out to a temporary file.

    In parallel, the generator spawns slurm jobs that run the ligand_adder
    script to add additional ligands and dabble. As each of those finishes
    it checks how many dabbled files are out there, and if they're all there
    the generation field in the config file will be updated.
    """

    def __init__(self, prodfiles, clusters, msm, scores, config, **kwargs):
        """
        Args:
            prodfiles (list of str): Trajectories, in order
            clusters (pickle thing): Cluster assigned to each frame
            scores (list): Hub score of all clusters, in order
            config (ConfigParser): Config file with other info
            criteria (str): Resampling criteria. Allowed: 'hub_scores',
                'populations'
        """
        # Input options
        self.prodfiles = prodfiles
        self.clusters = clusters
        self.scores = scores
        self.msm = msm

        # Config file options
        self.config = config
        self.nreps = self.config.getint("model", "samplers")
        self.lignames = self.config["system"]["ligands"].split(',')
        self.nligs = self.config.getint("system", "num_ligands")

        # Resampling criteria
        self.criteria = kwargs.get("criteria", "hub_scores")
        if self.criteria not in ["hub_scores", "populations", "counts"]:
            raise ValueError("Invalid resampling criteria: %s" % self.criteria)

        # Internal variables
        self.outdir = os.path.join(os.path.abspath(self.config["system"]["rootdir"]),
                                   "systems",
                                   str(self.config.getint("production", "generation")+1))
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

        if self.criteria == "hub_scores":
            self.bulk_clusters = np.argsort(self.scores)[-3:]
        elif self.criteria == "populations":
            self.bulk_clusters = np.argsort(self.msm.populations_)[-3:]
        elif self.criteria == "counts":
            self.counts = self._get_counts()
            self.bulk_clusters = self.counts[-3:]


    #==========================================================================

    def run(self):
        """
        This does all the work. Put it in another method to save
        an indentation level.
        """
        with VmdSilencer(output=os.path.join(self.config["system"]["rootdir"],
                                             "systems",
                                             str(self.config.getint("production", "generation")+1),
                                             "generator.log")):
            self.generate_initial_ensemble()

    #==========================================================================

    def generate_initial_ensemble(self):
        """
        Generates an initial ensemble of frames where one ligand
        is picked based on its hub score. This method alone does
        the exact same thing as the previous code, but will be
        extended to alter the positions of the other ligands
        to correspond to other resampled clusters.
        """

        # Choose which clusters to resample
        if self.criteria == "hub_scores":
            selected = list(self.msm.inverse_transform(self.scores.argsort())[0])
        elif self.criteria == "populations":
            selected = list(self.msm.inverse_transform(self.msm.populations_.argsort())[0])
        elif self.criteria == "counts":
            selected = self.counts # From cluster labels, so no need for transform

        # If enforcing sample region, this will be non-None
        selstr = self.config.get("model", "sampleregion")

        # Go through all of our resamplers until the correct number
        # have been done. Use cluster list as a refilling queue.
        molids = []
        while len(molids) != self.nreps:
            # Pick a cluster from the cluster queue
            clust = selected.pop(0)
            selected.append(clust)

            # Generate a frame with this cluster and check resampled ligand
            # is in the restricted selection area, if present
            mid = -1
            failures = 0
            while mid == -1 and failures <= 3:
                mid = self.pick_random_frame(clust)
                if not self._validate_frame(mid, selstr):
                    print("        Attempt fail for cluster: %d" % clust)
                    failures += 1
                    molecule.delete(mid)
                    mid = -1

            # If this cluster didn't work, continue to the next one
            if mid == -1:
                print("  Failed to select cluster: %d" % clust)
                continue

            # Otherwise, add it to our list of sampled clusters
            molids.append(mid)

            # Save and start adding job for this
            self.save_initial_molecule(molid=mid,
                                       sampled=clust,
                                       repindex=len(molids))

    #==========================================================================

    def _validate_frame(self, molid, sampleregion=None):
        """
        Validates a randomly selected frame to ensure that it is a valid
        protein/ligand combination. Checks that the ligand is in the
        selection area (if there is one for this run), and also that it
        is in the box.

        Args:
            molid (int): VMD molecule ID to check
            sampleregion (str): atom selection for selection area

        Returns:
            bool: True if this molecule is valid
        """
        # Check ligand is in the selection area if present
        if sampleregion is not None and \
           not len(atomsel("(%s) and (user 1.0) and "
                           "(same fragment as resname %s)" \
                           % (sampleregion,
                              " ".join(self.lignames)), molid=molid)):
                return False

        # Check ligand isn't out of the box, which can happen sometimes
        # with solvent ligands. This can result in some weird dabbled
        # systems that later crash
        box = [float(x)/2. for x in self.config["dabble"]["dimensions"].split(',')]
        ligsel = atomsel("user 1.0 and same fragment as resname %s"
                         % " ".join(self.lignames), molid=molid)

        #pylint: disable=too-many-boolean-expressions
        if max(ligsel.get("x")) > box[0] or \
           min(ligsel.get("x")) < -box[0] or \
           max(ligsel.get("y")) > box[1] or \
           min(ligsel.get("y")) < -box[1] or \
           max(ligsel.get("z")) > box[2] or \
           min(ligsel.get("z")) < -box[2]:
            print("   Rejected out of box")
            return False
        #pylint: enable=too-many-boolean-expressions

        return True

    #==========================================================================

    def save_initial_molecule(self, molid, sampled, repindex):
        """
        Saves a single molecule added thing. Names it according to
        the index and the sampled cluster.
        """
        # Write out to the output directory indicating the cluster
        # that was selected in the file name
        outnam = os.path.join(self.outdir, "%d_init_%d.mae" % (repindex, sampled))

        # Since we operate only off stripped prmtops, this should contain
        # the full system minus any dabble additions
        atomsel("all").write("mae", outnam)

        # Start a slurm job that will add additional ligands and dabble
        old_dir = os.getcwd()
        os.chdir(self.config["system"]["rootdir"])
        subprocess.call(["sbatch",
                         "--time=2:00:00",
                         "--partition=rondror",
                         "--tasks=2",
                         "--cpus-per-task=1",
                         "--mem=8GB",
                         "--job-name=%s_dabbler_G%d-%d"
                         % (self.config["system"]["jobname"],
                            self.config.getint("production",
                                               "generation")+1,
                            repindex),
                         "--export=INPUT=%s,CONFIG=%s"
                         % (outnam, os.path.join(self.config["system"]["rootdir"],
                                                 "sampler.cfg")),
                         "--output=%s" % os.path.join(self.outdir,
                                                      "dabble_%d.log" % repindex),
                         "--open-mode=append",
                         os.path.join(self.config["system"]["scriptdir"],
                                      "ligand_adder.py")
                         ])
        os.chdir(old_dir)

    #==========================================================================

    def pick_random_frame(self, cluster):
        """
        Gets a frame corresponding to a random cluster and loads it
        into a VMD molecule. Sets the x,y,z coordinate of all non-selected
        ligands to 0.
        except for the protein and selected ligand.

        Args:
            cluster (int): The cluster to sample
        Returns:
            (int): VMD molecule ID corresponding to this frame
        """
        fileindex, frameindex, ligidx = utils.get_frame(cluster,
                                                        dataset=self.clusters,
                                                        nligs=self.nligs)

        # Generate a VMD molecule containing this frame
        fid = utils.load_trajectory(filename=self.prodfiles[fileindex],
                                    config=self.config,
                                    frame=frameindex)

        # Identify selected ligand and change that field
        ligids = sorted(set(atomsel("resname %s"
                                    % " ".join(self.lignames),
                                    molid=fid).get("residue")))
        if not len(ligids):
            raise ValueError("No ligands found with resname(s) %s "
                             "in files %s"
                             % (self.lignames, molecule.get_filenames(fid)))

        # Use the user field to indicate which ligand is good.
        # This is necessary for checking if it's in the selection area later
        atomsel("same fragment as residue %d"
                % ligids[ligidx]).set("user", 1.0)

        # Set the xyz coordinates of all non-selected ligands to 0
        # We don't use the user field since that's not preserved
        # when saving to a mae file.
        ligids.remove(ligids[ligidx])
        if len(ligids):
            badligs = atomsel("same fragment as residue %s"
                              % " ".join([str(_) for _ in ligids]),
                              fid)
            badligs.set("x", 0.0)
            badligs.set("y", 0.0)
            badligs.set("z", 0.0)

        return fid

    #==========================================================================

    def _get_counts(self):
        def _count_one(self, num):
            return np.sum([len(np.where(c == num)[0]) for c in self.clusters])

        nclust = self.config.getint("model", "macrostates")
        return sorted(range(nclust), key=lambda x: _count_one(self, x))

    #==========================================================================
