from flask import Flask, request, render_template
from imports import pd 
from imports import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
# Make sure to run train.py first to generate this file!
try:
    model = joblib.load("gradient_model.pkl")
except FileNotFoundError:
    print("Model file not found. Please run train.py to create gradient_model.pkl")
    exit()

# Load the dataset to get the unique values for dropdowns and create stable mappings
try:
    data = pd.read_csv("data/adult.data", header=None)
    data.columns = [
        "age", "workclass", "fn", "education", "education-num", "marital",
        "occupation", "relationship", "race", "sex", "capital-gain",
        "capital-loss", "hours-per-week", "native-country", "income"
    ]
except FileNotFoundError:
    print("Data file not found. Make sure adult.data is in the 'data' directory.")
    exit()

# Define which columns are categorical for our dropdowns
categorical_columns = [
    "workclass", "education", "marital", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Create a dictionary to hold the unique values for each categorical column
dropdown_options = {col: data[col].unique().tolist() for col in categorical_columns}

# Create the stable mapping dictionaries from the training data
# This is crucial for consistent predictions
mappings = {}
for col in categorical_columns:
    unique_vals = data[col].unique()
    mappings[col] = {val: i for i, val in enumerate(unique_vals)}


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    # When the user submits the form, it's a POST request
    if request.method == 'POST':
        # Get the user inputs from the form
        user_inputs = request.form.to_dict()

        # Convert numerical fields from string to number
        numerical_inputs = {
            "age": int(user_inputs.get('age', 0)),
            "fn": int(user_inputs.get('fn', 0)),
            "education-num": int(user_inputs.get('education-num', 0)),
            "capital-gain": int(user_inputs.get('capital-gain', 0)),
            "capital-loss": int(user_inputs.get('capital-loss', 0)),
            "hours-per-week": int(user_inputs.get('hours-per-week', 0)),
        }
        
        # Create a pandas DataFrame from the inputs
        input_df = pd.DataFrame([user_inputs])

        # Apply the pre-saved mappings to the categorical columns
        for col in categorical_columns:
            input_df[col] = input_df[col].map(mappings[col])

        # Ensure all columns are in the correct order for the model
        # This is the order from your training script
        model_columns = [
            "age", "workclass", "fn", "education", "education-num", "marital",
            "occupation", "relationship", "race", "sex", "capital-gain",
            "capital-loss", "hours-per-week", "native-country"
        ]
        input_df = input_df[model_columns]

        # Make prediction
        prediction = model.predict(input_df)[0] # Get the first (and only) prediction

        # Format the output text
        if prediction == 0:
            result = "<= 50K"
        else: # Assumes the other value is 1
            result = "> 50K"
            
        prediction_text = f"Predicted Income: {result}"
        
        # Render the page again, this time with the prediction
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            options=dropdown_options,
            # Pass back the user's selections to keep the form populated
            selected_values=user_inputs
        )

    # When the user first visits the page, it's a GET request
    else:
        return render_template(
            'index.html',
            prediction_text="",
            options=dropdown_options,
            selected_values={} # Empty on first load
        )

if __name__ == '__main__':
    # The host='0.0.0.0' makes the app accessible from your local network
    app.run(debug=True, host='0.0.0.0')