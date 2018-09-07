"""
Methods for checking on the progress of an adaptive sampling run
"""
import datetime
import os
import subprocess
import time
from glob import glob
from configparser import ConfigParser
#pylint: disable=invalid-name

#==============================================================================

runstatuses = "PENDING RUNNING SUSPENDED COMPLETING COMPLETED".split(' ')

#==============================================================================

class AdaptiveSamplingChecker(object):
    """
    Checks the currently running production jobs.
    If they aren't running enough, start more.
    If they've all run enough, submit the MSM controller job
    The MSM controller job kills all the production jobs at its start.
    """

    def __init__(self, configfile):
        # Sanity check arguments
        if not os.path.isfile(configfile):
            raise FileNotFoundError("No config file: %s" % configfile)

        # Open the config file and figure out the production directory
        self.configfile = configfile
        self.config = ConfigParser(interpolation=None)
        self.config.read(configfile)

        print("Checking on: %s" % self.config["system"]["jobname"])
        print(datetime.datetime.now().strftime("%d %m %Y %H:%M"))

        # IF active project, figure out other stuff
        self.thedir = os.path.abspath(self.config["system"]["RootDir"])
        self.generation = self.config.getint("production", "generation")
        self.prodir = os.path.join(self.thedir, "production",
                                   str(self.generation))

        self.repnums = sorted(int(n.split('/')[-1])
                              for n in os.listdir(self.prodir)
                              if os.path.isdir(os.path.join(self.prodir, n)))

    #===========================================================================

    def check_in_queue(self, name):
        """
        Checks if a job with given name is queued or running.

        Args:
            name (str): Job name

        Returns:
            (int): Number of jobs of that name in queue
        """
        chk = subprocess.Popen(["squeue",
                                "-h",
                                "-u", os.environ.get("USER", "rbetz"),
                                "-o", "%F %T", # %F gets base ID for jobarray
                                "-n", name,
                               ],
                               stdout=subprocess.PIPE)
        # Will be of format ["jobid state", "jobid state"] etc
        chkres = chk.stdout.read().decode("utf-8").strip().split('\n')
        return len([c for c in chkres if len(c.split(" ")) > 1 and
                    c.split(" ")[1] in runstatuses])

    #===========================================================================

    def sbatch_job(self, job, cwd, dependency="singleton", extra_args=None):
        """
        Submits a job to the scheduler.

        Args:
            job (str): Path to the file to be sbatched
            cwd (str): Working directory from which to submit
            dependency (str): Dependency information
            extra_args (list of str): Additional arguments to sbatch

        Returns:
            (int): Job ID assigned
        """
        command = [
            "sbatch",
            "--dependency=%s" % dependency, # Add dependencies
        ]
        if extra_args:
            command.extend(extra_args)
        command.append(job)

        batcher = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE)
        return batcher.stdout.read().decode("utf-8").strip().split(' ')[-1]

    #===========================================================================

    def resubmit_job(self, rep, still_dabbling):
        """
        Resubmits a job that's running. Detects what step (minimization,
        equilibration, production) we're on and sets up dependencies

        Args:
            rep (int): Replicate ID to resubmit
            still_dabbling (bool): If generation numbers are out of sync
        """
        gen = self.generation + 1 if still_dabbling else self.generation
        depline = "singleton"
        eqdepline = "singleton"
        prefix = "%s-G%d-r%d" % (self.config.get("system", "jobname"),
                                 gen, rep)

        # Check minimization completed, if not, resubmit it and equilibration
        mindir = os.path.join(self.thedir, "minimization",
                              str(gen), str(rep))

        if not os.path.isfile(os.path.join(mindir, "min3.rst")) and \
                not self.check_in_queue("%s_min_%d" % (prefix, rep)):
            print("  Restarting minimization for %s" %  prefix)
            minid = self.sbatch_job("PROTOCOL.sh", cwd=mindir,
                                    dependency="singleton")
            eqdepline += ",afterok:%s" % minid


        # Check if equilibration is complete or running. If not, restart it
        eqdir = os.path.join(self.thedir, "equilibration",
                             str(gen), str(rep))
        if not os.path.isfile(os.path.join(eqdir, "Eq_5.rst")):
            # Do not queue more production jobs if waiting on equilibration
            if self.check_in_queue("%s_eq_%d" % (prefix, rep)):
                return
            else:
                print("  Restarting equilibration for %s" % prefix)
                eqid = self.sbatch_job("PROTOCOL.sh", cwd=eqdir,
                                       dependency=eqdepline)
                depline += ",afterok:%s" % eqid


        # Check for queued production jobs. Otherwise add a new jobarray
        if not self.check_in_queue("%s_prod_%d" % (prefix, rep)):

            nruns = self.config.getint("production", "submissions")
            self.sbatch_job("PROTOCOL.sh",
                            cwd=os.path.join(self.prodir, str(rep)),
                            dependency=depline,
                            extra_args=["--array=1-%d%%1" % nruns])

            print("  Started more of %s" % prefix)
            time.sleep(1.)

    #===========================================================================

    def get_simulation_time(self, rep):
        """
        Returns the amount of simulation time a given replicate has completed,
        by parsing the mdinfo.

        Args:
            rep (int): Replicate to check

        Returns:
            (float) Amount of time, or 0.0 if no mdinfo or an error was found
        """
        stime = 0.0
        mdinfo = os.path.join(self.prodir, str(rep), "mdinfo")
        if os.path.isfile(mdinfo):
            grep = subprocess.Popen(["grep",
                                     "-oh",
                                     r"TIME(PS) = \+[[:digit:]]\+\.[[:digit:]]\+",
                                     mdinfo],
                                    stdout=subprocess.PIPE)
            output = grep.stdout.read().decode('utf-8').strip()
            if output:
                try:
                    stime = float(output.split(' ')[-1])/1000.
                except ValueError:
                    print("Problem getting simtime. Str='%s'" % output)
                    stime = 0.0

        return stime

    #===========================================================================

    def spawn_msm(self):
        """
        Starts a Markov state model job, and kills all production jobs

        Args:
            config (ConfigParser): Configuration object for this run

        Returns:
            (int) Slurm job ID of the MSM
        """

        # Stop all production jobs based on name
        deljobnames = ["%s-G%d-r%d_prod_%d" % (self.config["system"]["jobname"],
                                               self.generation,
                                               x, x) for x in self.repnums]
        deljobnames.extend(["%s_dabbler_G%d-%d"
                            % (self.config["system"]["jobname"],
                               self.generation+1,
                               x) for x in self.repnums])

        for djob in deljobnames:
            subprocess.call(["scancel", "-u", "rbetz", "-n", djob])

        jobname = self.config.get("system", "jobname")
        model_args = [
            "--export=CONFIG=%s" % self.configfile,
            "--qos=normal",
            "--job-name=%s_G%d" % (jobname, self.generation+1),
            "--output=%s" % os.path.join(self.thedir, "production",
                                         "%s_G%d"
                                         % (jobname, self.generation)),
        ]
        return self.sbatch_job(os.path.join(self.config["system"]["scriptdir"],
                                            "controller.py"),
                               cwd=self.prodir,
                               extra_args=model_args)

    #===========================================================================

    def check_run(self):
        """
        Checks the progress of a run. Starts new jobs as necessary depending
        on what's currently happening. May start MSM, continue simulations,
        restart dabblers, etc.

        Args:
            configfile (str): Path to configuration file for the run
        Returns:
            (bool): True if more jobs were submitted
        """

    #===========================================================================

    def check(self):
        """
        Does all checks and looks at simulations

        Returns:
            (bool): If a new MSM was started
        """
        # Check if MSM is running, and if so, just stop here
        if self.check_msm_running():
            return False

        # Now check individual replicates
        dabbling = self.check_dabblers()
        if self.check_simulations(dabbling):
            return False

        # Update config file with new generation
        # Was incremented in-memory when MSM spawned
        with open(self.configfile, 'w') as cfgfile:
            self.config.write(cfgfile)

        return True

    #===========================================================================

    def check_msm_running(self):
        """
        Check if MSM job is running

        Returns:
            (bool): If job is still running
        """

        # Check if MSM job is running and if so, return
        check = subprocess.Popen("squeue -u rbetz -o %%j | grep %s_G%d | wc -l"
                                 % (self.config["system"]["jobname"],
                                    self.generation+1),
                                 shell=True, stdout=subprocess.PIPE)
        status = int(check.stdout.read().decode("utf-8").strip())
        return status > 0

    #===========================================================================

    def check_dabblers(self):
        """
        Checks if dabblers needs to be re-invoked

        Returns:
            (list of int): Replicate indices that were re-dabbled
        """

        sysdir = os.path.join(self.thedir, "systems", str(self.generation+1))

        dabbling = []
        for repidx in range(1, self.config.getint("model", "samplers")+1):

            # Check if dabbler is running
            if self.check_in_queue("%s_dabbler_G%d-%d"
                                   % (self.config["system"]["jobname"],
                                      self.generation+1,
                                      repidx)) > 0:
                print("  Waiting on dabbler %d" % repidx)
                dabbling.append(repidx)
                continue

            # Check if dabbler needs to be restarted
            inpfile = glob(os.path.join(sysdir, "%d_init_*.mae" % repidx))
            if not (os.path.isfile(os.path.join(sysdir, "%d.inpcrd" % repidx))) \
                    and len(inpfile) == 1:
                print("   Restarting dabbler %d" % repidx)
                dabbling.append(repidx)

                subprocess.call(["sbatch",
                                 "--time=2:00:00",
                                 "--partition=rondror",
                                 "--qos=normal",
                                 "--tasks=2",
                                 "--cpus-per-task=1",
                                 "--mem=8GB",
                                 "--job-name=%s_dabbler_G%d-%d"
                                 % (self.config["system"]["jobname"],
                                    self.generation+1,
                                    repidx),
                                 "--export=INPUT=%s,CONFIG=%s"
                                 % (inpfile[0], self.configfile),
                                 "--output=%s" % os.path.join(sysdir,
                                                              "dabble_%d.log"
                                                              % repidx),
                                 "--open-mode=append",
                                 os.path.join(self.config["system"]["scriptdir"],
                                              "ligand_adder.py")
                                ])
        return dabbling

    #===========================================================================

    def check_simulations(self, dabbling):
        """
        Checks if production simulations need to be restarted due to time
        lengths and the current amount of simulations in the queue

        Args:
            dabbling (list of int): Replicates that are being re-dabbled
                and so shouldn't be considered for simulation restart

        Returns:
            (bool) If simulations were complete
        """
        # Loop through each replicate and find if we need to submit more job
        num_done = 0
        to_resubmit = [] # Keep track of them and only resubmit if we don't MSM
        runlength = self.config.getfloat("production", "runlength")
        runpercent = self.config.getfloat("production", "runpercent")

        # If we're dabbling, the generation of running jobs is one higher
        # than what's in the config file
        thisgen = self.generation + 1 if dabbling else self.generation

        # Check all replicates that aren't dabbling
        for r in [x for x in self.repnums if x not in dabbling]:

            # Get the current simulation time from the mdinfo file
            simtime = self.get_simulation_time(r)
            print("  Sampler: %d Time: %f Runl: %f" % (r, simtime, runlength))
            if simtime < runlength:
                to_resubmit.append(r)

            # Reached correct length, killall current jobs to free up queue
            else:
                num_done += 1
                print("  Reached time, killed G%d-R%d" % (thisgen, r))
                subprocess.call(["scancel",
                                 "-u", "rbetz",
                                 "-n", "%s-G%d-r%d_prod_%d"
                                 % (self.config["system"]["jobname"],
                                    thisgen, r, r)
                                ])

        # If all jobs met the minimum length criteria, spawn a MSM builder job
        if not dabbling and num_done >= runpercent*len(self.repnums):
            print("Done with generation %d!" % self.generation)
            self.config["model"]["JobID"] = self.spawn_msm()
            print("   MSM ID: %s" % self.config["model"]["JobID"])
            return False

        # Otherwise, resubmit unfinished jobs
        self.config["model"]["JobID"] = ""
        print(" Gen %d Not finished yet..." % thisgen)
        for r in to_resubmit:
            self.resubmit_job(r, still_dabbling=len(dabbling)>0)
        return True

    #===========================================================================
