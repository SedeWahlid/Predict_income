import imports
import func

#read data 
data = imports.pd.read_csv("data/adult.data")
#add columns to the data
data.columns = [
    "age",
    "workclass", 
    "fn", 
    "education", 
    "education-num", 
    "marital", 
    "occupation", 
    "relationship", 
    "race", 
    "sex", 
    "capital-gain", 
    "capital-loss", 
    "hours-per-week", 
    "native-country", 
    "income"]

# keys for changing strings to numbers 
columns_to_map = [
    "workclass", "education", "marital", "occupation", 
    "relationship", "race", "sex", "native-country", "income"
]

# function call to change the columns from strings to values (int)
data = func.mapping(columns_to_map, data)

# splitting the data
x_train,x_test, y_train,y_test = func.split_data(data)

# train data and save the model 
model = func.train_and_save_grad_boost(x_train,y_train)
pred = model.predict(x_test)
print(imports.classification_report(y_test, pred))