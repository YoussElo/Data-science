import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# Load the Twibot 22 dataset
data = pd.read_csv("twibot_22.csv")

# Preprocess the data by cleaning and formatting it
data = data.drop_duplicates()
data = data.dropna()
data = data[data["bot"] != "unverified"]

# Convert the "bot" column to a binary label
data["bot"] = data["bot"].map({"bot": 1, "human": 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("bot", axis=1), data["bot"], test_size=0.2)

# Scale the data using min-max normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose an appropriate machine learning model for the task
# Here, we'll use three different models: GCN, GAT, and RGCN

# First, we'll build a GCN model
gcn_model = GCN(input_dim=X_train.shape[1], hidden_dim=128, output_dim=2)

# Next, we'll build a GAT model
gat_model = GAT(input_dim=X_train.shape[1], hidden_dim=128, output_dim=2, n_heads=8)

# Finally, we'll build an RGCN model
rgcn_model = RGCN(input_dim=X_train.shape[1], hidden_dim=128, output_dim=2, num_rels=3, num_bases=30)

# Train the models on the training set
gcn_model.fit(X_train, y_train)
gat_model.fit(X_train, y_train)
rgcn_model.fit(X_train, y_train)

# Evaluate the models on the testing set
gcn_score = gcn_model.score(X_test, y_test)
gat_score = gat_model.score(X_test, y_test)
rgcn_score = rgcn_model.score(X_test, y_test)
# Calculate the accuracy and F1 score for each model
gcn_accuracy = accuracy_score(y_test, gcn_model.predict(X_test))
gcn_f1 = f1_score(y_test, gcn_model.predict(X_test))
gat_accuracy = accuracy_score(y_test, gat_model.predict(X_test))
gat_f1 = f1_score(y_test, gat_model.predict(X_test))
rgcn_accuracy = accuracy_score(y_test, rgcn_model.predict(X_test))
rgcn_f1 = f1_score(y_test, rgcn_model.predict(X_test))

# Print the accuracy and F1 score for each model
print("GCN accuracy:", gcn_accuracy)
print("GCN F1 score:", gcn_f1)
print("GAT accuracy:", gat_accuracy)
print("GAT F1 score:", gat_f1)
print("RGCN accuracy:", rgcn_accuracy)
print("RGCN F1 score:", rgcn_f1)

# Print the scores
print("GCN score:", gcn_score)
print("GAT score:", gat_score)
print("RGCN score:", rgcn_score)

