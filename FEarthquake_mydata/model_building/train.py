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

api = HfApi()

# 1. Set the library verbosity level (DEBUG, INFO, WARNING, etc.)
transformers.logging.set_verbosity_info()

# 2. Get the core Hugging Face logger
hf_logger = transformers.logging.get_logger()

# 3. Create a file handler to write to 'huggingface.log'
file_handler = logging.FileHandler("huggingface.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 4. Add the handler to the logger
hf_logger.addHandler(file_handler)


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
from sklearn.ensemble import RandomForestRegressor
gb_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

#clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# 5. Train (fit) the model on the training data
#clf.fit(Xtrain, ytrain)
# Define hyperparameter grid
param_grid = {
    'gradientboostingclassifier__n_estimators': [75, 100, 125],
    'gradientboostingclassifier__max_depth': [2, 3, 4],
    'gradientboostingclassifier__subsample': [0.5, 0.6]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, gb_model)

# Grid search with cross-validation
#grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
#grid_search.fit(Xtrain, ytrain)
gb_model.fit(Xtrain, ytrain)

# Best model
#best_model = grid_search.best_estimator_
#print("Best Params:\n", grid_search.best_params_)

# Predict on training set
#y_pred_train = best_model.predict(Xtrain)
y_pred_train = gb_model.predict(Xtrain)
#y_pred_train = clf.predict(Xtrain)

# Predict on test set
#y_pred_test = best_model.predict(Xtest)
y_pred_test = gb_model.predict(Xtest)
 
# Evaluation
print("\nTraining Classification Report:")
 
# 5. Evaluate the model
mae = mean_absolute_error(ytest, y_pred_test)
mse = mean_squared_error(ytest, y_pred_test)
r2 = r2_score(ytest, y_pred_test)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared ($R^2$) Score: {r2:.4f}")

#print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
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
