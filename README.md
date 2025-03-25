# IPL First Innings Score Predictor

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![ML Library](https://img.shields.io/badge/ML_Library-Scikit_Learn-orange)
![RMSE](https://img.shields.io/badge/RMSE-20.08-green)
![Deployment](https://img.shields.io/badge/Deployment-Heroku-purple)

A machine learning web application that predicts the first innings score of Indian Premier League (IPL) cricket matches based on the current match situation.

![Prediction Interface](https://github.com/faseehahmed26/IPL_First_Innings_Score/blob/main/Images/after.png?raw=true)

## Features

- Predicts likely final score based on current match statistics
- Considers batting team, bowling team, current score, wickets, overs, and recent performance
- Uses optimized Ridge Regression model for accurate predictions
- Provides results within ~20 runs RMSE of actual scores
- Simple, user-friendly web interface

## Technology Used

- **Machine Learning**: Ridge Regression, Lasso Regression
- **Data Processing**: Pandas, NumPy
- **Model Training**: Scikit-learn, GridSearchCV
- **Web Framework**: Flask
- **Deployment**: Heroku
- **Frontend**: HTML, CSS, Bootstrap

## Model Development

The system uses a Ridge Regression model trained on IPL match data from 2008-2016, with matches from 2017 onwards used for testing. The model was developed through the following process:

1. **Data Preprocessing**:
   - Filtering for consistent teams across IPL seasons
   - Converting categorical features using one-hot encoding
   - Temporal train-test split based on match years

2. **Feature Engineering**:
   - Current runs, wickets, and overs played
   - Recent performance (runs and wickets in last 5 overs)
   - Team matchups through one-hot encoded variables

3. **Model Selection**:
   - Compared Ridge and Lasso regression models
   - Hyperparameter tuning using GridSearchCV
   - Ridge Regression with alpha=40 performed best

4. **Results**:
   - Mean Absolute Error: 14.86 runs
   - Root Mean Square Error: 20.08 runs

## Installation and Usage

```bash
# Clone repository
git clone https://github.com/faseehahmed26/IPL_First_Innings_Score.git
cd IPL_First_Innings_Score

# Install requirements
pip install -r requirements.txt

# Run application
python app.py
