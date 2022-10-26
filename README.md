# Disaster Response Pipeline Project
### Summary: 
The primary goal of this project is to develop a model which can be exposed via a flask application which provides a easy way to categorize the disaster related messages into one of the 36 categories.

### Files:
- data\process_data.py - ETL pipeline to process, clean the data and save it into database.
- models\train_classifier.py - ML pipeline to train the model based on cleaned data and save the model into .pkl file
- models\classifier.pkl - Saved model parameters
- app\run.py - Flask application with endpoints (loads test data and saved model) 
- requirements.txt - List of all python libraries used for the whole application. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
