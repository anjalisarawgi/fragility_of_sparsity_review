# The Fragilty of Sparsity : Review and Analysis
Implementation and testing of models from the paper 'The Fragility of Sparsity'.

This repository aims to reproduce the results of the paper with 2 different datasets to check the relevance of the results. The main goal is to understand the generalizability of the results of the paper. 

If you use this work, please cite the following paper:

Michal Kolesár, Ulrich K. Müller, Sebastian T. Roelsgaard (2024). *The Fragility of Sparsity*. arXiv. https://arxiv.org/abs/2311.02299


## Overview
The paper "The Fragility of Sparsity" explores the stability and reliability of sparse models, which are often used in economics, statistics, social sciences as well as machine learning for feature selection and model simplification. This repository seeks to replicate the key experiments from the paper and assess their relevance across different datasets.


### Datasets

The experiments are conducted using the following three datasets:

1. **Communities and Crime Unnormalized**: This dataset includes socio-economic data from communities across the United States and is used to predict crime rates. More information and access to the dataset can be found [here](https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized).
2. **LaLonde Dataset**: This dataset is used in causal inference studies and includes data from an observational study on the effect of a job training program on real earnings. The dataset and an example of its use can be found [here](https://www.pywhy.org/dowhy/v0.9/example_notebooks/dowhy_lalonde_example.html).


### Key Goals

- **Reproduce the Results**: Implement the models and experiments from the paper to verify the reported outcomes.
- **Test on Multiple Datasets**: Evaluate the models on two different datasets to test the robustness and generalizability of the results.
- **Analyze Fragility**: Understand how sensitive the sparsity models are to changes in the dataset and other parameters.

## Getting Started

### Prerequisites
To run the code in this repository, you will need the following:

- Python 3.8 or higher
- Scikit-Learn

- ### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/anjalisarawgi/fragility_of_sparsity_review.git
cd fragility_of_sparsity_review
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

To run the experiments, execute the following command:
```bash
python main.py
```
The results of the experiments will be stored in the results/ directory. You can analyze these results to draw comparisons with the findings in the paper.


### Additional Notes

- **Lasso Alpha Tuning**: To apply different Lasso alpha values (including setting to LassoCV), you need to manually modify the `models.py` file. The default values are set in the `main.py` file, but adjustments can be made by editing `src/data/models/model.py` directly.

- **Custom Method/Data Adjustments**: To change the code according to different methods, data sets, or dimensions, you must manually update the section under `if __name__ == "__main__"` in the `main.py` file.

## Citation
If you find this work useful in your research, please cite the following paper:

```bash
@article{kolesar2024fragility,
  title={The Fragility of Sparsity},
  author={Michal Kolesár and Ulrich K. Müller and Sebastian T. Roelsgaard},
  journal={arXiv preprint arXiv:2311.02299},
  year={2024}
}
```

## Contributions

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any improvements or suggestions.









