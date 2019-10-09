#%%
from plumbum import SshMachine
from git import Repo
from plumbum.cmd import git
import numpy as np
from fabric import Connection

def expandListsInExperiments(experimentSuite, fields_to_expand, conditions, label):
    experiments = experimentSuite['experiments']
    while fields_to_expand:
        new_experiments = []
        field_to_expand = fields_to_expand.pop()

        for exp in experiments:
            new_exps = []
            for value in exp[field_to_expand]:
                new_exp = exp.copy()
                new_exp[field_to_expand] = value
                new_exps.append(new_exp)
            new_experiments += new_exps
        experiments = new_experiments
        experimentSuite['experiments'] = new_experiments
    
    # Replace random_seed=None with randint
    for exp in experiments:
        if exp['random_seed'] is None:
            exp['random_seed'] = np.random.randint(10000)
    return experimentSuite


def updateRepo(servername, project_dir, repo_ssh_string, branch):
    """
    Helper function to pull newest commits to remote repo.
    """
    remote = SshMachine(servername)
    r_git = remote['git']
    home_dir = remote.cwd
    # Update repository
    print("Updating repo...", end='')
    with remote.cwd(home_dir / project_dir):
        r_git('fetch', '-q', 'origin')
        r_git('checkout', branch)
        r_git('reset', '--hard', 'origin/{}'.format(branch), '-q')

    # Check that we have the same git hash remote than local
    with remote.cwd(home_dir / project_dir):
        r_head = r_git('rev-parse', 'HEAD')
    l_head = git('rev-parse', 'HEAD')
    assert l_head == r_head, "Local git hash != pushed git hash. Did you forget to push changes?"
    print("Repo updated")
    remote.close()


def startExperimentSet(experimentSuite,
                       free_gpus,
                       project_dir,
                       repo_ssh_string,
                       repo_path="../../",
                       branch='master',
                       ):
    print("Starting experiments...")
    experiments = experimentSuite['experiments']
    label = experimentSuite['label']
    conditions = experimentSuite['conditions']
    for gpu, exp in zip(free_gpus[:len(experiments)], experiments):
        """
        gpu should be a tuple (servername(str), gpu-nr(int))
        """

        # home_dir = remote.cwd
        assert not "logdir" in exp.keys(), "'logdir' should not be in the list of changed parameters"

        condition_str = "_".join([exp[key] for key in conditions.split(",")])
        cmd = "cd myfivo && sh ./run.sh {} run_fivo.py ".format(gpu[1])
        cmd += " ".join(["--{}={}".format(key, exp[key]) for key in exp])
        cmd += " --logdir=/home/t-maigl/results/{}_{}".format(label, condition_str)
        cmd += " &> {}_{}.log".format(label, condition_str)

        updateRepo(gpu[0], project_dir, repo_ssh_string, branch)
        print("CMD: {}".format(cmd))
        print("Running command on remote...")
        c = Connection(gpu[0])
        c.run(cmd)
        print(".done")

experimentSuite = { 
    'experiments': [{ 
        'random_seed': [None]*1,
        'mode': 'train',
        'model': 'vrnn',
        'bound': 'elbo',
        'batch_size': 4,
        'latent_size': 32,
        'num_samples': 4,
        'learning_rate': 0.0003,
        'dataset_path': './pianorolls/jsb.pkl',
        'dataset_type': 'pianoroll',
        'proposal_type': ['filtering', 'filtering_cutq', 'filtering_cutp', 'prior_conditioned', 'priorl_conditioned']
        }], 
    'conditions': 'proposal_type', 
    'label': '0304_elbo'}

free_gpus = [('vm0', 0), ('vm0', 1), ('vm0', 2), ('vm0', 3), ('vm1', 0), ('vm1', 1), ('vm1', 2), ('vm1', 3)]
# free_gpus = [('vm1', 1), ('vm1', 2), ('vm1', 3)]
free_gpus = [('vm1', 0)]

if experimentSuite['conditions'] == '': to_expand = ['random_seed']
else: to_expand = ['random_seed']+experimentSuite['conditions'].split(",")
experimentSuite = expandListsInExperiments(experimentSuite,
                                           to_expand,
                                           conditions=experimentSuite['conditions'],
                                           label=experimentSuite['label'])

experimentSuite['experiments'] = experimentSuite['experiments'][4:]

# print (experimentSuite['experiments'][0])

print("Looking to start {} experiments".format(len(experimentSuite['experiments'])))

#%%

startExperimentSet(experimentSuite=experimentSuite,
                   free_gpus=free_gpus, 
                   project_dir='myfivo', 
                   repo_ssh_string='git@github.com:maximilianigl/myfivo.git', 
                   repo_path="./", 
                   branch='master')

#%%

# from fabric import Connection
# c = Connection('vm')
# r0 = c.run('cd myfivo', hide=True)
# print(r0)
# result = c.run('ls', hide=True)
# print(result)

