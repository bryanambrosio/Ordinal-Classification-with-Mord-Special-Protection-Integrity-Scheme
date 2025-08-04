# Ordinal Classification for Special Protection Scheme Using Mord

**Author:** Bryan Ambrósio  
**Affiliations:**  
- Universidade Federal de Santa Catarina (UFSC)  
- Laboratório de Planejamento de Sistemas Elétricos (LabPlan)  
- INESC Brasil – Instituto de Engenharia de Sistemas e Computadores

---

This project implements an ordinal classification pipeline aimed at predicting the minimum number of generator units to be disconnected in critical power system scenarios. The classification respects the natural ordering of classes, which is crucial for making more meaningful and operationally relevant predictions.

## Overview

We use the Python library **`mord`**, which provides models specifically designed for ordinal regression problems. These models differ from traditional classifiers by explicitly accounting for the order of the classes rather than treating them as unrelated categories.

## Key Components

- **Data preprocessing:**  
  - Loaded measurements and system variables from a realistic dataset.  
  - Cleaned and filtered data based on quality criteria.  
  - Scaled features with standardization to improve model convergence.

- **Handling class imbalance:**  
  - Applied **SMOTE** oversampling on the training data to balance underrepresented classes.  
  - Used **sample weighting** to emphasize minority classes during training.

- **Models evaluated:**  
  - `OrdinalRidge` — Ridge regression adapted for ordinal targets.  
  - `LogisticIT` and `LogisticAT` — Logistic regression models tailored for ordinal data with different threshold schemes.

- **Hyperparameter tuning:**  
  - Performed grid search over regularization strength (`alpha`) aiming to minimize false negatives on training data, as these represent critical misclassifications in our application.

- **Evaluation and visualization:**  
  - Calculated detailed per-class metrics including false negatives, false positives, true positives, and true negatives.  
  - Visualized classification results projected into 2D space using PCA, highlighting false negatives and positives distinctly for training and test sets.

## Conclusions

- Ordinal classifiers from `mord` effectively leverage the ordered nature of the problem, reducing severe misclassifications.  
- Combining SMOTE oversampling with sample weighting improves model fairness across classes.  
- Hyperparameter tuning focused on minimizing false negatives helps adapt the model to operational priorities.  
- Visualizations provide intuitive insight into the model’s behavior and error distribution.

## Future Work

- Explore alternative ordinal models or neural approaches for further performance gains.  
- Incorporate time-series or domain-specific features to enrich the dataset.  
- Develop real-time evaluation tools for deployment in operational environments.

---

This repository serves as a foundation for applying ordinal classification techniques in power system protection and can be extended for other applications requiring ordered decision-making.
