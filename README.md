Formula 1 Podium Prediction
This project uses a Random Forest Classifier to predict podium finishes (top 3 positions) for Formula 1 drivers in the 2025 Chinese Grand Prix, based on historical data from the 2022–2024 seasons. The model leverages race results, team performance, and driver statistics to estimate podium probabilities.
Table of Contents

Overview
Dataset
Features
Requirements
Installation
Usage
Results
Contributing
License

Overview
The project processes Formula 1 race data to train a machine learning model that predicts the likelihood of drivers finishing on the podium. Key steps include:

Data preprocessing and merging from multiple F1 datasets.
Feature engineering to capture team and driver performance metrics.
Training a Random Forest Classifier with balanced class weights.
Evaluating model performance using accuracy, cross-validation, classification reports, and confusion matrices.
Predicting podium probabilities for the 2025 Chinese Grand Prix.

Dataset
The project uses datasets from the Ergast F1 API or similar sources, stored as CSV files:

results.csv: Race results, including driver positions and points.
races.csv: Race metadata, such as year and circuit ID.
constructors.csv: Team information.
drivers.csv: Driver details, including names.

Note: Datasets are not included in this repository due to size and licensing. You can download them from a reliable source or use the provided file structure.
File Structure
├── Datasets/
│   ├── results.csv
│   ├── races.csv
│   ├── constructors.csv
│   ├── drivers.csv
├── f1_podium_prediction.py
├── README.md

Features
The model uses the following features:

Team Features:
avg_grid: Average starting grid position per year.
total_points: Total constructor points per year.
avg_position: Average finishing position per year.


Driver Features:
driver_points: Total points earned by the driver per year.
past_podiums: Number of podiums achieved by the driver per year.
races_participated: Number of races the driver participated in per year.


Race Feature:
circuitId: Encoded ID of the race circuit (e.g., Shanghai = 17).



The target variable is podium (1 for top 3 finish, 0 otherwise).
Requirements

Python 3.8+
Libraries:
pandas
numpy
scikit-learn



Installation

Clone the repository:git clone https://github.com/your-username/f1-podium-prediction.git
cd f1-podium-prediction


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt

Or manually install:pip install pandas numpy scikit-learn


Place the required CSV files in the Datasets/ directory.

Usage

Ensure the Datasets/ folder contains results.csv, races.csv, constructors.csv, and drivers.csv.
Run the script:python f1_podium_prediction.py


The script will:
Load and preprocess the data.
Train the Random Forest model.
Output model performance metrics (accuracy, cross-validation scores, classification report, confusion matrix).
Display podium probabilities for all drivers in the 2025 Chinese Grand Prix, sorted by likelihood.
Highlight the top predicted podium contender.



Example Output
Model Accuracy: 0.92
Cross-Validation Accuracy (5-fold): 0.91 (±0.02)

Classification Report:
              precision    recall  f1-score   support
Non-Podium     0.94      0.97      0.95       500
Podium         0.75      0.60      0.67        50
...

Podium Probabilities for 2025 Chinese Grand Prix:
------------------------------------------------------------
Red Bull Racing   | Max Verstappen    |  92.50%
McLaren           | Lando Norris      |  85.30%
...
------------------------------------------------------------
Predicted Top Podium Contender: Max Verstappen (Red Bull Racing) (Probability: 92.50%)

Results

The model achieves high accuracy (typically ~90%) due to the class imbalance (podiums are rare).
Cross-validation ensures robust performance.
The confusion matrix and classification report provide insights into precision and recall, especially for the minority class (podiums).
Predictions for the 2025 Chinese Grand Prix are based on 2024 team and driver performance, assuming similar conditions.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/new-feature).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature/new-feature).
Open a pull request.

Report issues or suggest improvements via the Issues tab.
License
This project is licensed under the MIT License. See the LICENSE file for details.
