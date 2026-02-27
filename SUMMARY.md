`Project Overview`

In this project, I built a fully reproducible end-to-end Machine Learning pipeline using Random Forest and DVC (Data Version Control). The goal was not only to train a model but to structure the entire workflow following industry best practices, including data versioning, experiment tracking, and reproducibility.

`Understanding Reproducible ML Pipelines`

One of the most important learnings was understanding how to build a modular ML pipeline using DVC stages:
Data loading
Preprocessing
Training
Evaluation
Feature importance
Error analysis

Using dvc repro, I learned how DVC automatically detects changes in dependencies and re-runs only necessary stages. This introduced me to dependency graphs and deterministic pipelines, which are critical in production ML systems.

`Hyperparameter Management and Experiment Tracking`

By defining hyperparameters inside params.yaml, I learned:
Why separating configuration from code improves maintainability.
How DVC tracks parameter changes using -p flags.
How to compare experiments using dvc metrics show and dvc metrics diff.
I observed how changing n_estimators and max_depth impacted performance, and how DVC cache prevents unnecessary retraining.
This helped me understand structured experimentation instead of manually rerunning scripts.

`Random Forest Insights`

Through experimentation, I learned:

Random Forest does not require feature scaling because it is tree-based.
Increasing the number of trees improves model stability but may not significantly improve performance after convergence.
Feature importance is calculated based on total impurity (Gini) reduction across trees.
Impurity-based importance has limitations, especially with correlated features.

`Model Evaluation and Error Analysis`

Using confusion matrix and classification metrics, I learned:
The importance of minimizing false negatives in medical datasets.
Why recall is critical when detecting malignant tumors.
How ROC-AUC measures overall discrimination ability.
How to interpret precision, recall, and F1-score together.
Error analysis helped me think beyond accuracy and understand real-world implications of predictions.

`Conclusion`

This project strengthened my understanding of both Machine Learning modeling and MLOps practices. I learned how to design reproducible pipelines, manage experiments systematically, and analyze model behavior in a structured way.
It gave me practical exposure to production-style ML development rather than just academic model training.