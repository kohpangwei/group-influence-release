# On the Accuracy of Influence Functions for Measuring Group Effects

This code replicates the experiments from the following paper:

> Pang Wei Koh, Kai-Siang Ang, Hubert H. K. Teo and Percy Liang
>
> [On the Accuracy of Influence Functions for Measuring Group
> Effects](https://arxiv.org/abs/1905.13289)

We have a reproducible, executable, and Dockerized version of the paper on
[CodaLab](https://worksheets.codalab.org/worksheets/0xfed2ae0b9e5b44b7a1af8096365592a5).

## Abstract

Influence functions estimate the effect of removing particular training points
on a model without needing to retrain it. They are based on a first-order
approximation that is accurate for small changes in the model, and so are
commonly used for studying the effect of individual points in large datasets.
However, we often want to study the effects of large groups of training points,
e.g., to diagnose batch effect or apportion credit between different data
sources. Removing such large groups can result in significant changes to the
model. Are influence functions still accurate in this setting? In this paper,
we find that across many different types of groups and in a range of real-world
datasets, the influence of a group correlates surprisingly well with its actual
effect, even if the absolute and relative error can be large. Our theoretical
analysis shows that such correlation arises under certain settings but need not
hold in general, indicating that real-world datasets have particular properties
that keep the influence approximation well-behaved. 

## Prerequisites

- Python 2.7
- NumPy 1.16.5
- SciPy 1.2.2
- matplotlib 2.2.4
- seaborn 0.9.0
- jupyter 1.0.0
- scikit-learn 0.20.4
- Pandas 0.24.2
- Spacy 2.0.0
- Tensorflow 1.14.0 (GPU or CPU)
- Tensorflow Probability 0.7.0

### Local installation

You can install the Python dependencies locally using `pip` and
`requirements.txt`.  Note that our `requirements.txt` includes the GPU version
of Tensorflow.  It is also possible to run the code on the CPU using
Tensorflow.  We highly recommend isolating these dependencies with `virtualenv`
or some other Python environment manager.
```
virtualenv -p python2.7 ~/influence/
source ~/influence/bin/activate
pip install -r requirements.txt
```

Then, pre-download the Spacy model used for preprocessing the Enron dataset.
```
python -m spacy download en_core_web_sm
```

### Docker

We also provide a `Dockerfile` for the execution environment. A pre-built image
`bbbert/influence:5` can be found in
[DockerHub](https://hub.docker.com/r/bbbert/influence/tags).
`bbbert/influence:5` is the image used in the CodaLab worksheet.

## Note on folder structure and CodaLab

The code maintains and expects a particular folder structure, parameterized by two base directories: a data directory, and an output directory to store the
result of experiment runs. All scripts in `scripts/` that use these directories will expect the `--data-dir` and `--out-dir` CLI arguments, which default
to `[repository_root]/data/` and `[repository_root]/output/` respectively. Then, we will download and preprocess datasets and save experiment outputs
in the following structure:
```
.
├─── data/
│    ├─── hospital/                                                               (A data directory, not necessarily a dataset ID)
│    ├─── spam/
│    └─── ...
└─── output/
     ├─── ss_logreg/                                                              (An experiment ID)
     │    ├─── spam_ihvp-explicit_seed-0_sizes-0.0025-0.25_num-10_rel-reg-1e-07/  (One particular run of the experiment)
     │    └─── ...
     ├─── counterexamples/
     └─── ...
```

Notice that each experiment in `experiments/` has a fixed experiment ID, and different runs of the same experiment are stored as subfolders
in `[out_dir]/[experiment_id]/[run_id]/`. The run ID is a string defined by the experiment that is meant to contain all the relevant
parameters for the run.

## Usage

All experiments and operations are performed by executing the scripts in
`scripts/`.  The code expects the repository root to be in the Python path. We
recommend appending the repository root to the `PYTHONPATH` environment
variable, and executing the scripts from the repository root. For example, to
download and preprocess all the datasets, run
```
export PYTHONPATH=${PYTHONPATH}:$(pwd)
python scripts/load_all_datasets.py
```

## CodaLab

For convenience, the `codalab/` folder contains scripts used in our
[CodaLab](https://worksheets.codalab.org/worksheets/0xfed2ae0b9e5b44b7a1af8096365592a5)
worksheet. The sequence of runs that generates the worksheet can be found in
`codalab/codalab_run.sh`, except for the MultiNLI dataset, which we have
preprocessed and uploaded as a [CodaLab bundle](https://worksheets.codalab.org/bundles/0x5258e3771b974983abfb10bfa75e207d).
