# Assignment-3--breast-cancer
Machine learning Asssignment
This assignment uses "Breast cancer.csv" dataset which contains features of cell nuclei extracted from digitized images of breast tissues.The id column represents record ID, and “diagnosis” column indicates the diagnosis outcome (“M” – Malignant, “B” – Benign). The other columns are numeric and represent features extracted or calculated from the cell nuclei.
In this assignment, a number of ML algorithms are implemented to predict cancerous cells using features avaliable in the data-set.
Subsequently, PCA (Principal Component Analysis) is used for dimensionality reduction on the dataset and a logistic regression model is then built using a limited number of PCAs.

In this assignment, data is partitioned into 70% for training and 30% for testing. A seven-fold cross validation  is used to train the model with ML algorithms namely - Logistic Regression, Support Vector Machine, Random Forest, Neural Net and the model performance results (confusion matrix) are then use to determine which model has the best performance.
