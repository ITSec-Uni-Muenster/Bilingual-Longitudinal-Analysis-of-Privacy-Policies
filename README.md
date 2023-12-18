# Bilingual Longitudinal Analysis of Privacy-Policies

### NOTE: This repository is under construction. We are working on documenting the code and adding instructions. 

## Basic Requirements
We ran the code in Anaconda environments. Please find the .yml files in the repository and the accompanying description below. 

## Software Requirements
All experiments were conducted on Linux machines, which were either equipped with CentOS 7.9 or Ubuntu 20.04.

## Estimated Time and Storage Consumption
The experiments were performed on a high-performance cluster. The nodes used were equipped with either 128 cores and 512GB of RAM or 72 cores and 1.5 TB of RAM.
The determination of the number of topics (See Appendix H) using the ```ldatuning``` library took up to 7 days for each corpus. 

## Environment
The Anaconda software is available at [Anaconda.com](https://www.anaconda.com/download). Please download and install Anaconda on your Linux System. Then use the environment files to create the environments. 

### Set up the environment
There are two Anaconda environments which need to be installed using the provided .yml files. One is a Python 3.7 environment, which was used for the analysis. The number of topics in each corpus is determined by using the other environment, which is based on R 4.1.

### Experiments
The data, i.e., the privacy policies were collected over multiple time points from the end of 2017 to the beginning of 2023. The collected data from December 2017 up to November 2018 were used in [this publication](https://www.ndss-symposium.org/ndss-paper/we-value-your-privacy-now-take-some-cookies-measuring-the-gdprs-impact-on-web-privacy/) as well. For the collection of the data starting from December 2019, a privacy policy collection tool based on OpenWPM was used, which was complemented by a carefully crafted preprocessing module [published](https://petsymposium.org/popets/2021/popets-2021-0081.pdf) in 2021. All data were preprocessed and sanitized by using this module. Since 2021, this toolchain was further improved and is fully available [here](https://github.com/ITSec-Uni-Munster/Unifying-Privacy-Policy-Detection).

For the data analysis, we conducted keyness analysis, co-occurrence and bigram-depencency analysis, ``Do Not Sell My Personal Information'' analysis using regular expressions, and topic modeling over time using [BERTopic](https://maartengr.github.io/BERTopic/).
#### Experiment 1: Keyness Analysis
The keyness analysis was conducted using the code base of the Keyness Python library V0.25. This code base outputs the phrase, log-likelihood (LL), number of occurrences in the reference corpus (CC), and number of occurrences in the references corpus (RCC). We expanded this library to output the following further statistics:
- Bayesian Information Criterion (BIC)
- G2 effect size for the loglikelihood (ELL)
- Odds ratio (OR)
- Log ratio (LR)
- Relative risk (RR)
- Percentage difference (PercDiff)
- Difference coefficient (DiffC)
- Overuse oder underuse (WordUse)
- Expected value in the reference corpus (RCC.E)
- Expected value in the target corpus (CC.E)
- Total number of phrases in the target corpus (CT)
- Total number of phrases in the reference corpus (RCT)

Note that not all functions in this file were required for the analysis. 

Prior to to the Keyness analysis, tokenization of the lemmatized privacy policies was performed using the [SoMaJo](https://github.com/tsproisl/SoMaJo) and [SoMeWeTa](https://github.com/tsproisl/SoMeWeTa) libraries.

#### Experiment 2: Co-occurrence and Bigram-depencency Analysis
The code for this analysis was expanded from the [corpus_toolkit](https://github.com/kristopherkyle/corpus_toolkit/blob/master/corpus_toolkit/corpus_tools.py) library. Various improvements were mading including the addition of the Log Dice test, Chi2 test, Fisher's Exact test, the phi value, and the relative frequency per million. 

Note that not all functions in this file were required for the analysis. 


#### Experiment 3: ``Do Not Sell My Personal Information''Analysis


#### Experiment 4: Topic Modeling over Time with BERTopic


## Notes on Reusability
Unfortunately, we cannot share the collected privacy policies due to intellectual property rights. The code is still applicable to any diachronic corpus. 
