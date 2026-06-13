# for data manipulation
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

HF_username="hasilm1"
App_name="Earthquake_prediction"
Model_name="best_predict_earthquake_model_v1.joblib"

Xtrain_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/Xtrain.csv"                    # enter the Hugging Face username here
Xtest_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/Xtest.csv"                      # enter the Hugging Face username here
ytrain_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/ytrain.csv"                    # enter the Hugging Face username here
ytest_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/ytest.csv"                      # enter the Hugging Face username here

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

#id,time,year,month,day_of_year,hour,latitude,longitude,depth,mag
#year,month,day_of_year,hour,longitude,depth,mag

# scale numeric features
numeric_features = [
    'year',
    'month',
    'day_of_year',
    'hour',
    'longitude',
    'depth',
    'mag'
]


# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define GB model
#gb_model = GradientBoostingClassifier(random_state=42)
#gb_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

 
# 1. Define the hyperparameter search space
param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10,20,30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', 0.3]
}

# 2. Initialize the baseline regressor
#rf = RandomForestRegressor(n_estimators=100,random_state=42, n_jobs=-1)
rf = RandomForestRegressor(
    n_jobs=-1,             # Use all CPU cores (Crucial)
    n_estimators=100,      # Keep tree count reasonable
    max_depth=15,          # Stop trees from growing too deep
    min_samples_leaf=5,    # Stop splits early when data thins out
    max_features='sqrt',   # Scan fewer columns per split
    random_state=42
)

#model_pipeline = make_pipeline(preprocessor, rf)

# 3. Setup the randomized cross-validation search
rf_random = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_distributions, 
    n_iter=2,          # 50 Number of random combinations to try
    cv=3,               # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    random_state=42, 
    n_jobs=-1
)

rf_random.fit(Xtrain, ytrain)

# 5. Extract the optimized model
#gb_model = rf_random.best_estimator_
gb_model = rf_random
#print("Best Parameters Found:", rf_random.best_params_)

# Predict on training set
#y_pred_train = best_model.predict(Xtrain)
y_pred_train = gb_model.predict(Xtrain)
#y_pred_train = clf.predict(Xtrain)

# Predict on test set
#y_pred_test = best_model.predict(Xtest)
y_pred_test = gb_model.predict(Xtest)
 
# Evaluation
print("\nEvaluation Report:")
 
# 5. Evaluate the model
mae = mean_absolute_error(ytest, y_pred_test)
mse = mean_squared_error(ytest, y_pred_test)
r2 = r2_score(ytest, y_pred_test)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared ($R^2$) Score: {r2:.4f}")

#print(classification_report(ytrain, y_pred_train))

#print(classification_report(ytest, y_pred_test))

# Save best model
#joblib.dump(best_model, Model_name)
joblib.dump(gb_model, Model_name)

# Upload to Hugging Face
repo_id = str(HF_username)+"/"+str(App_name)                                         # enter the Hugging Face username here
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))
 
# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=Model_name,
    path_in_repo=Model_name,
    repo_id=repo_id,
    repo_type=repo_type,
)
