# FROD
The code and dataset for paper "Weakly-Supervised Outlier Detection With Fuzzy Rough Sets".

## Datasets
We use 16 public datasets to assess the model performances, including 2 nominal, 2 mixed, and 12 numerical datasets. The details of the datasets are provided in below table: 

| No |     Datasets     | # Objects | # Attributes | # Outlier | % Outlier |   Category   |  Data Type  |
|:--:|:----------------:|:---------:|:------------:|:---------:|:---------:|:------------:|:-----------:|
|  1 |    annthyroid    |    7200   |       6      |    534    |    7.4%   |  Healthcare  |  Numerical  |
|  2 |    Arrhythmia    |    452    |      279     |     66    |   14.6%   |    Medical   |    Mixed    |
|  3 | Cardiotocography |    2114   |      21      |    466    |   22.0%   |  Healthcare  |  Numerical  |
|  4 |    Ionosphere    |    351    |      32      |    126    |   35.9%   |  Oryctognosy |  Numerical  |
|  5 |    mammography   |   11183   |       6      |    260    |    2.3%   |  Healthcare  |  Numerical  |
|  6 |     Mushroom1    |    4429   |      22      |    221    |    5.0%   |    Botany    | Categorical |
|  7 |     Mushroom2    |    4781   |      22      |    573    |   12.0%   |    Botany    | Categorical |
|  8 |       musk       |    3062   |      166     |     97    |    3.2%   |   Chemistry  |  Numerical  |
|  9 |     optdigits    |    5216   |      64      |    150    |    2.9%   |     Image    |  Numerical  |
| 10 |    PageBlocks    |    5393   |      10      |    510    |    9.5%   |   Document   |  Numerical  |
| 11 |       Pima       |    768    |       8      |    268    |   34.9%   |  Healthcare  |  Numerical  |
| 12 |     satellite    |    6435   |      36      |    2036   |   31.6%   | Astronautics |  Numerical  |
| 13 |    satimage-2    |    5803   |      36      |     71    |    1.2%   | Astronautics |  Numerical  |
| 14 |       Sick       |    3613   |      29      |     72    |    2.0%   |    Medical   |    Mixed    |
| 15 |     SpamBase     |    4207   |      57      |    1679   |   39.9%   |   Document   |  Numerical  |
| 16 |      thyroid     |    3772   |       6      |     93    |    2.5%   |  Healthcare  |  Numerical  |

## Environment
* cudatoolkit=11.6.0
* numpy=1.23.5
* pandas=1.5.3
* python=3.8.16
* pytorch=1.12.1
* scikit-learn=1.2.0
* scipy=1.9.3
* torchaudio=0.12.1
* torchvision=0.13.1
