import imports
import func

# --- LOAD SAVED ARTIFACTS ---
print("Loading model and supporting files...")
try:
    model = imports.joblib.load("gradient_model.pkl")
    print("Files loaded successfully.")
except FileNotFoundError:
    print("Error: Model or mapping files not found.")
    print("Please run train.py first to generate these files.")
    exit()

# --- GET ARBITRARY USER INPUT ---
print("\nPlease provide the following details to predict income:")
user_input = {
    "age": int(input("Age: ")),
    "workclass": input("Workclass (e.g., Private, Self-emp-not-inc, Local-gov): "),
    "fn": int(input("Final Weight (fnlwgt): ")),
    "education": input("Education (e.g., Bachelors, HS-grad): "),
    "education-num": int(input("Education-num: ")),
    "marital": input("Marital Status (e.g., Married-civ-spouse, Never-married): "),
    "occupation": input("Occupation (e.g., Tech-support, Exec-managerial): "),
    "relationship": input("Relationship (e.g., Husband, Not-in-family): "),
    "race": input("Race (e.g., White, Black): "),
    "sex": input("Sex (e.g., Male, Female): "),
    "capital-gain": int(input("Capital Gain: ")),
    "capital-loss": int(input("Capital Loss: ")),
    "hours-per-week": int(input("Hours per Week: ")),
    "native-country": input("Native Country (e.g., United-States, Mexico): ")
}
# keys for changing strings to numbers 
columns_to_map = [
    "workclass", "education", "marital", "occupation", 
    "relationship", "race", "sex", "native-country"
]

# make user input to a pandas dataframe 
user_input = imports.pd.DataFrame([user_input])
# make user data with columns 
user_input.columns = [
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
    "native-country"]

# change strings to values 
user_input = func.mapping(columns_to_map,user_input)
# change NaN to -1 
user_input.fillna(-1)

# --- MAKE AND DISPLAY PREDICTION ---
try:
    prediction = func.predict_from_input(user_input, model)

    print("\n--- Prediction Result ---")
    if prediction == 0 :
        print(f"Predicted Income Category: >= 50K")
    elif prediction == 1:
        print(f"Predicted Income Category: < 50K")

except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")
    print("Please ensure your inputs are valid (e.g., 'Private' for workclass).")