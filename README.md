
## Scripts
### 1. `preeprocess.py`
This script handles data preprocessing tasks, including:
- Encoding symptoms, genetic data, and exercise information.
- Cleaning and formatting the data to ensure consistency.
- Preparing the data for training by creating necessary features.

### 2. `train.py`
This script is responsible for training the predictive model. It includes:
- Loading the preprocessed data.
- Building an LSTM model for predicting suitable exercises based on genetic information.
- Training the model and saving the trained model for future predictions.

### 3. `test.py`
This script is used to test and evaluate the performance of the trained model. It includes:
- Loading the trained model and test data.
- Generating predictions for the test dataset.
- Calculating accuracy metrics to assess the model's performance.

## Datasets
- **Disorders Folder**: Contains detailed information on various genetic disorders, symptoms, and their impact on physical health.
- **Exercises Dataset**: Lists exercises and the muscle groups they target.
- **Symptoms Dataset**: Links symptoms to relevant genetic disorders.
- **Genetic Data**: Contains genetic information of individuals, used for training and testing the model.

## How to Run the Project
1. Clone the repository to your local machine:<br>
  git clone https://github.com/Sameerbeedi/Genes-to-Gyms.git

2. Navigate to the project directory:<br>
  `cd Genes-to-Gyms`

 3.Run the preprocessing script to prepare the data:<br>
  `python preeprocess.py`

4. Train the model using the `train.py` script:<br>
  python train.py

5. Test the model's performance using the `test.py` script:<br>
  python test.py

## Dependencies
Ensure that the following Python libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`

Install dependencies using:
pip install -r requirements.txt

## Contribution
Feel free to contribute to the project by raising issues or submitting pull requests. Make sure to follow the coding guidelines and provide appropriate documentation for any new features or bug fixes.

