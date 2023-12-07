# Bilingual Longitudinal Analysis of Privacy-Policies

Paper title: A Bilingual Longitudinal Analysis of Privacy Policies Measuring the Impacts of the GDPR and the CCPA/CPRA

## Basic Requirements
We ran the code in Anaconda environments. Please find the .yml files in the repository and the accompanying description below. 

### Software Requirements
All experiments were conducted on Linux machines, which were either equipped with CentOS 7.9 or Ubuntu 20.04.

### Estimated Time and Storage Consumption
The experiments were performed on a high-performance cluster. The nodes used were equipped with either 128 cores and 512GB of RAM or 72 cores and 1.5 TB of RAM.
The determination of the number of topics (See Appendix XXX) using the ```ldatuning``` library took 4 to 7 days for each corpus. 

## Environment
Anaconda is available at https://www.anaconda.com/download. Please download and install Anaconda on your Linux System. Then use the 

### Set up the environment
There are two Anaconda environments which need to be installed using the provided .yml files. One is a Python 3.7 environment, which was used for the analysis. The number of topics in each corpus is determined by using the other environment, which is based on R 4.1.

### Experiments
List each experiment the reviewer has to execute. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes and how much space it consumes on disk. (approximately)
 - Which claim and results does it support, and how.

#### Experiment 1: Name
Provide a short explanation of the experiment and expected results.
Describe thoroughly the steps to perform the experiment and to collect and organize the results as expected from your paper.
Use code segments to support the reviewers, e.g.,
```bash
python experiment_1.py
```
#### Experiment 2: Name
...

#### Experiment 3: Name
...

## Notes on Reusability
Unfortunately, we cannot share the collected privacy policies due to intellectual property rights. The code should still be applicable to any diachronic corpus. 
