# Modeling citation worthiness by using attention-based bidirectional long short-term memory networks and interpretable models

This is the repository contains the code and data for the Scientometrics paper: Modeling citation worthiness by using attention-based bidirectional long short-term memory networks and interpretable models

# Data

ACL‑ARC dataset: please refer to [Bonab et al., 2018](http://doi.org/10.1145/3209978.3210162) for details. We downloaded a copy of dataset, adjusted some fields. You can download it from Figshare: [10.6084/m9.figshare.12573872](https://doi.org/10.6084/m9.figshare.12573872).

PMOA-CITE dataset: please download 1M sentences from Figshare: [10.6084/m9.figshare.12547574](https://doi.org/10.6084/m9.figshare.12547574)

PMOA-CITE and ACL-ARC combined: please download it from Figshare:  [10.6084/m9.figshare.12573974](https://doi.org/10.6084/m9.figshare.12573974)


# Dependencies
The code requires the following packages:
- allennlp==0.9.0
- scikit-learn==0.21.2


# Run the experiments

All the experiments configuration files are located in cite-worthiness/experiments folder, to run an experiment:
1. Please find the fields train_data_path, validation_data_path and test_data_path in each jsonnet file, and change the value to the path where you store the datasets mentioned above.
2. Find the cuda_device field, change it to -1 if you're using a CPU, otherwise the CUDA device number.
3. Run the command:
```
allennlp train /path/to/experiment/configuration/jsonnet/file -s ../path/to/serialization/dir/  --include-package citation_worthiness
```
Please refer to allennlp documentation for the use of `train` command


# How to cite
If you use the dataset and code on this repo, please cite our work: Modeling citation worthiness by using attention-based bidirectional long short-term memory networks and interpretable models.

```
@Article{Zeng2020,
    author={Zeng, Tong and Acuna, Daniel E.},
    title={Modeling citation worthiness by using attention-based bidirectional long short-term memory networks and interpretable models},
    journal={Scientometrics},
    year={2020},
    month={Jul},
    day={01},
    volume={124},
    number={1},
    pages={399-428},
    issn={1588-2861},
    publisher = {Springer International Publishing},
    doi={10.1007/s11192-020-03421-9},
    url={https://doi.org/10.1007/s11192-020-03421-9}
}
```
#  Science of Science and Computational Discovery Lab
The datasets and code are developed in the [Science of Science and Computational Discovery Lab](https://scienceofscience.org/) in the School of Information Studies, Syracuse University.

# License
The code in this repo use the MIT license.