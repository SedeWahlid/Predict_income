import imports

#splitting data into train, test data 
def split_data(data):
    x = data[[i for i in data if not i == "income"]]
    y = data["income"]
    X_train, X_test, y_train, y_test = imports.train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test


#changing data to numbers by creating a dict to map to the column
def mapping (col_str, data):
    for str in col_str:
        temp = data[str].unique()
        diction = {}
        k = 0
        for i in temp:
            diction.update({i:k})
            k+= 1
        data[str] = data[str].replace(diction)
    return data

# Trains the Gradient Boosting model, saves it to a file, and returns it.
def train_and_save_grad_boost(X_train, y_train, model_path="gradient_model.pkl"):
    print("Training the Gradient Boosting model...")
    model = imports.GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    imports.joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

# predicting arbitary input 
def predict_from_input(user_input , model):
    pred = model.predict(user_input)
    return pred