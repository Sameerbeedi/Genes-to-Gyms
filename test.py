import ast
import pandas as pd
from collections import Counter
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model('network.h5', custom_objects={'mse': 'mean_squared_error'})

# Load the CSV data
df = pd.read_csv('encoded_training_symptoms.csv')

# Define the feature and exercise columns
feature_cols = ['Encoded_Symptom', 'Encoded_Genetic']
exercise_cols = [col for col in df.columns if col.startswith('Encoded_') and col not in feature_cols]

# Convert string representations of lists into actual lists
for col in exercise_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Function to preprocess the input for prediction
def preprocess_input(symptom, genetic):
    symptom_encoded = df[df['Symptom'] == symptom]['Encoded_Symptom'].iloc[0]
    genetic_encoded = df[df['Name'] == genetic]['Encoded_Genetic'].iloc[0]
    X = np.array([[symptom_encoded, genetic_encoded]])

    exercise_values = np.stack(df[exercise_cols].apply(lambda row: np.concatenate(row.values), axis=1).values)
    X = np.hstack((X, exercise_values[0].reshape(1, -1)))

    X_scaled = scaler.transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_reshaped

# Function to get exercises based on filtered dataset
def get_exercises_for_symptom_genetic(symptom, genetic):
    # Filter the dataset based on the specific symptom and genetic disorder
    filtered_df = df[(df['Symptom'] == symptom) & (df['Name'] == genetic)]

    if filtered_df.empty:
        print("No data available for this symptom and genetic disorder combination.")
        return {}

    # Calculate the frequency of each exercise for this specific combination
    exercise_frequencies = {col: 0 for col in exercise_cols}
    for _, row in filtered_df.iterrows():
        for col in exercise_cols:
            exercise_frequencies[col] += sum(row[col])

    return exercise_frequencies

# Decode the most frequent exercises for each muscle group
def decode_predictions(exercise_frequencies):
    decoded_exercises = {}
    for muscle_group, muscle_group_exercises in unique_exercises_by_muscle_group.items():
        # Calculate frequency of each exercise
        exercise_counts = Counter({exercise: exercise_frequencies.get(f'Encoded_{exercise}', 0) for exercise in muscle_group_exercises})

        # Sort exercises based on frequency (descending order)
        sorted_exercises = sorted(exercise_counts.items(), key=lambda x: x[1], reverse=True)

        # Select the most frequent exercise
        most_frequent_exercise = sorted_exercises[0][0] if sorted_exercises else None

        decoded_exercises[muscle_group] = most_frequent_exercise

    return decoded_exercises

# Predict exercises for a given symptom and genetic information
def predict_exercises(symptom, genetic):
    exercise_frequencies = get_exercises_for_symptom_genetic(symptom, genetic)
    if not exercise_frequencies:
        return

    # Decode the predictions to return the most frequent exercise for each muscle group
    exercises = decode_predictions(exercise_frequencies)
    return exercises

# Unique exercises by muscle group
unique_exercises_by_muscle_group = {
    'Back': [df['Back'].iloc[i] for i in range(len(df['Encoded_Back'].iloc[0])) if any(df['Encoded_Back'].apply(lambda x: x[i] == 1.0))],
    'Chest': [df['Chest'].iloc[i] for i in range(len(df['Encoded_Chest'].iloc[0])) if any(df['Encoded_Chest'].apply(lambda x: x[i] == 1.0))],
    'Triceps': [df['Triceps'].iloc[i] for i in range(len(df['Encoded_Triceps'].iloc[0])) if any(df['Encoded_Triceps'].apply(lambda x: x[i] == 1.0))],
    'Bicep/Forearm': [df['Bicep/Forearm'].iloc[i] for i in range(len(df['Encoded_Bicep/Forearm'].iloc[0])) if any(df['Encoded_Bicep/Forearm'].apply(lambda x: x[i] == 1.0))],
    'Quadriceps': [df['Quadriceps'].iloc[i] for i in range(len(df['Encoded_Quadriceps'].iloc[0])) if any(df['Encoded_Quadriceps'].apply(lambda x: x[i] == 1.0))],
    'Hamstrings/Glutes': [df['Hamstrings/Glutes'].iloc[i] for i in range(len(df['Encoded_Hamstrings/Glutes'].iloc[0])) if any(df['Encoded_Hamstrings/Glutes'].apply(lambda x: x[i] == 1.0))],
    'Calf/Abductor': [df['Calf/Abductor'].iloc[i] for i in range(len(df['Encoded_Calf/Abductor'].iloc[0])) if any(df['Encoded_Calf/Abductor'].apply(lambda x: x[i] == 1.0))],
    'Core/Ab': [df['Core/Ab'].iloc[i] for i in range(len(df['Encoded_Core/Ab'].iloc[0])) if any(df['Encoded_Core/Ab'].apply(lambda x: x[i] == 1.0))],
    'Shoulder': [df['Shoulder'].iloc[i] for i in range(len(df['Encoded_Shoulder'].iloc[0])) if any(df['Encoded_Shoulder'].apply(lambda x: x[i] == 1.0))],
    'Mobility': [df['Mobility'].iloc[i] for i in range(len(df['Encoded_Mobility'].iloc[0])) if any(df['Encoded_Mobility'].apply(lambda x: x[i] == 1.0))],
}

# Function to get the most frequent exercise for a specific symptom and genetic disorder
def get_most_frequent_exercise(symptom, genetic):
    predicted_exercises = predict_exercises(symptom, genetic)
    if predicted_exercises:
        for muscle_group, exercise in predicted_exercises.items():
            if exercise:
                print(f"The most frequent exercise for {muscle_group} is: {exercise}")

# Example usage
symptom = 'Balancing issues'  # Replace with the actual symptom
genetic = "Huntington's"  # Replace with the actual genetic information
get_most_frequent_exercise(symptom, genetic)
