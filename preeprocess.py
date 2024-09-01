import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the CSV data
df = pd.read_csv('training_symptoms.csv')

# Encode the symptoms and genetic names separately
symptom_label_encoder = LabelEncoder()
genetic_label_encoder = LabelEncoder()

# Fit encoders on symptom data
symptom_label_encoder.fit(df['Symptom'])

# Fit encoders on genetic data
genetic_label_encoder.fit(df['Name'])

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.3, random_state=42)



# Update the training and testing sets with the encoded names
train['symptom_name_encoded'] = symptom_label_encoder.transform(train['Symptom'])
test['symptom_name_encoded'] = symptom_label_encoder.transform(test['Symptom'])
train['genetic_name_encoded'] = genetic_label_encoder.transform(train['Name'])
test['genetic_name_encoded'] = genetic_label_encoder.transform(test['Name'])

# Function to extract symptoms from a column
def extract_symptom(df, column):
    symptoms = set()
    df[column].dropna().apply(lambda x: symptoms.update([symptom.strip() for symptom in x.split(',')]))
    return list(symptoms)

def extract_genetic(df, column):
    genetics = set()
    df[column].dropna().apply(lambda x: genetics.update([genetic.strip() for genetic in x.split(',')]))
    return list(genetics)

# Extract the list of symptoms
symptom_list = extract_symptom(df, 'Symptom')
print("Symptoms list:", symptom_list)

genetic_list = extract_genetic(df, 'Name')
print("Genetics list:", genetic_list)

# Create a mapping of symptom names to numerical values
symptom_mapping = dict(zip(symptom_label_encoder.classes_, symptom_label_encoder.transform(symptom_label_encoder.classes_)))
print("Symptom Name to Numerical Value Mapping:", symptom_mapping)

genetic_mapping = dict(zip(genetic_label_encoder.classes_, genetic_label_encoder.transform(genetic_label_encoder.classes_)))
print("Genetic Name to Numerical Value Mapping:", genetic_mapping)

# Function to encode exercises for a given list
def encode_exercises(exercise_list, unique_exercises, max_exercises=9):
    encoded_data = np.zeros(len(unique_exercises))
    if pd.isna(exercise_list) or exercise_list == '':
        return encoded_data
    exercises = [exercise.strip() for exercise in exercise_list.split(',') if exercise.strip() in unique_exercises]
    num_exercises = min(len(exercises), max_exercises)
    selected_exercises = np.random.choice(exercises, num_exercises, replace=False) if len(exercises) > num_exercises else exercises
    for exercise in selected_exercises:
        encoded_data[unique_exercises.index(exercise)] = 1
    return encoded_data

# Function to identify unique exercises for each muscle group based on symptoms and ensure exactly 9 exercises per group
def identify_exercises_by_symptom(df, symptom_col, muscle_group_cols):
    unique_exercises_by_symptom = {symptom: {col: set() for col in muscle_group_cols} for symptom in df[symptom_col].unique()}
    for symptom in unique_exercises_by_symptom:
        symptom_df = df[df[symptom_col] == symptom]
        for col in muscle_group_cols:
            if symptom_df[col].dtype == 'object':  # assuming exercises are stored as strings
                exercises = symptom_df[col].apply(lambda x: [exercise.strip() for exercise in x.split(',')]).explode().dropna()
                unique_exercises = exercises.value_counts().index.tolist()[:9]
                unique_exercises_by_symptom[symptom][col].update(unique_exercises)
    return unique_exercises_by_symptom

# Columns
symptom_col = 'Symptom'
non_muscle_group_columns = ['Name', 'Age', 'Gender', symptom_col]
muscle_group_cols = [col for col in df.columns if col not in non_muscle_group_columns]

# Identify unique exercises for each symptom
unique_exercises_by_symptom = identify_exercises_by_symptom(df, symptom_col, muscle_group_cols)

# Print the number of unique exercises for each muscle group under each symptom
for symptom, muscle_groups in unique_exercises_by_symptom.items():
    print(f'Symptom: {symptom}')
    for muscle_group, exercises in muscle_groups.items():
        print(f'  Muscle Group: {muscle_group} - Number of Exercises: {len(exercises)}')

# Loop through each muscle group column, extract unique exercises, and encode them
for col in muscle_group_cols:
    df['Encoded_' + col] = df.apply(
        lambda row: encode_exercises(row[col], list(unique_exercises_by_symptom[row[symptom_col]][col])),
        axis=1
    )

# Prepare the final DataFrame
final_df = df.copy()  # Create a copy to avoid SettingWithCopyWarning

# Add encoded muscle group columns to the final DataFrame using .loc
for col in muscle_group_cols:
    final_df.loc[:, 'Encoded_' + col] = final_df['Encoded_' + col].apply(lambda x: list(x))

# Encode the symptoms and genetic names and add as new columns
final_df['Encoded_Symptom'] = symptom_label_encoder.transform(final_df['Symptom'])
final_df['Encoded_Genetic'] = genetic_label_encoder.transform(final_df['Name'])

# Print the resulting dataframe for verification
print(final_df.head())

# Save the final DataFrame to a new CSV file
final_df.to_csv('encoded_training_symptoms.csv', index=False)
