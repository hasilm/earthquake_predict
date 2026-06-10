# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

Folder_name="FEarthquake_mydata"
HF_username="hasilm1"
App_name="Earthquake_prediction"
Column_name='latitude'
Column_name2='longitude'
Test_size=0.2
Random_state=42
data_filename="earthquakes.csv"

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
print(os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/"+str(data_filename)                  # enter the Hugging Face username here
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

target_col = str(Column_name)

# Split into X (features) and y (target)
X = df.drop(columns=[target_col,'id','time'])
X = X.dropna()

y = df[target_col,'id','time']

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=Test_size, random_state=Random_state)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]


for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=str(HF_username)+"/"+str(App_name),                                           # enter the Hugging Face username here
        repo_type="dataset",
    )
