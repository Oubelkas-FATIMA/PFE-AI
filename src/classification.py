from config.config import  CROP_MAPPING

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import pandas as pd


class DataPreprocessor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def remove_band_10(self):
        '''
        Removes all columns containing 'Band_10' in their name.
        '''
        colonnes_a_supprimer = self.df.filter(regex='Band_10').columns
        self.df.drop(columns=colonnes_a_supprimer, inplace=True, axis=1)

    def mean_df(self):
        '''
        Calculates the mean of values for each Field_Id after removing the 'Crop_Type' column.
        '''
        self.df.drop('Crop_Type',  inplace=True)
        self.df = self.df.groupby('Field_Id').mean().reset_index()
    
    def mapping_crop_id(self):
        '''
        Maps Crop_Id_Ne to corresponding crop names according to CROP_MAPPING.
        '''
        self.crop_mapping = CROP_MAPPING
        def map_crop_id(self, crop_id):
            return self.crop_mapping.get(crop_id, 'Unknown')
        self.df['Crop_Type'] = self.df['Crop_Id_Ne'].apply(map_crop_id)





    def remove_and_merge_classes(self):
        '''
        Merges certain crop classes and removes others.
        '''
        
        self.df.loc[self.df['Crop_Id_Ne'].isin([3, 7]), 'Crop_Type'] = 'Grass&Vacant'
        self.df.loc[self.df['Crop_Id_Ne'].isin([3, 7]), 'Crop_Id_Ne'] = 10
        self.df = self.df[~self.df['Crop_Id_Ne'].isin([9, 2, 3, 7])]
        

    def vineyard_intercrop(self):
        '''
        Balances data for vineyard and intercrop classes.
        '''
        
        classes_inclure = [6, 8, 9]
        df_filtered = self.df[self.df['Crop_Id_Ne'].isin(classes_inclure)]
        df1 = df_filtered.drop('Crop_Type', axis=1)
        mean_df = df1.groupby('Field_Id').mean().reset_index()
        value_counts = mean_df['Crop_Id_Ne'].value_counts()

        samples_8 = value_counts[9] * 3
        df_filtered_balanced = mean_df[mean_df['Crop_Id_Ne'].isin([6, 9])]
        df_class_8 = mean_df[mean_df['Crop_Id_Ne'] == 8].sample(n=samples_8, replace=True)
        df_b = pd.concat([df_filtered_balanced, df_class_8])
        balanced_df =  df_filtered[df_filtered['Field_Id'].isin(df_b['Field_Id'])]
        self.df = balanced_df

        

    def split_and_shuffle_data(self):
        '''
        Splits the data into training and test sets, then shuffles them.
        Output:
            train_df_shuffled (DataFrame): Shuffled training data
            test_df_shuffled (DataFrame): Shuffled test data
        '''
        classes = self.df['Crop_Id_Ne'].unique().tolist()
        df_z = self.df[['Field_Id', 'Crop_Id_Ne']].drop_duplicates()
        train_data = []
        test_data = []
        for classe in classes:
            data_classe = df_z[df_z['Crop_Id_Ne'] == classe]
            train_classe, test_classe = train_test_split(data_classe, test_size=0.3)
            train_data.append(train_classe)
            test_data.append(test_classe)
        train_data_concat = pd.concat(train_data)
        test_data_concat = pd.concat(test_data)
        train_df = self.df[(self.df['Crop_Id_Ne'].isin(train_data_concat['Crop_Id_Ne'])) & (self.df['Field_Id'].isin(train_data_concat['Field_Id']))]
        test_df = self.df[(self.df['Crop_Id_Ne'].isin(test_data_concat['Crop_Id_Ne'])) & (self.df['Field_Id'].isin(test_data_concat['Field_Id']))]
        train_df_shuffled = shuffle(train_df, random_state=1).reset_index(drop=True)
        test_df_shuffled = shuffle(test_df, random_state=1).reset_index(drop=True)
        return train_df_shuffled, test_df_shuffled
    

    def add_predictions_to_test_data(self, test_predictions, test_df_shuffled):
        '''
        Adds predictions to the test data.
        Input:
            test_predictions (array): Model predictions
            test_df_shuffled (DataFrame): Shuffled test data
        Output:
            df_r (DataFrame): DataFrame with Field_Id and predictions
        '''
        df_r = pd.DataFrame({'Field_Id': test_df_shuffled['Field_Id']})
        df_r['test_predictions'] = test_predictions
        return df_r
    



    def process_predictions(self, df_r):
        '''
        Processes predictions to obtain percentages by Field_Id.
        Input:
            df_r (DataFrame): DataFrame with Field_Id and predictions
        Output:
            df_percentage (DataFrame): DataFrame with prediction percentages by Field_Id
        '''
        dfs = []
        duplicated_field_ids = df_r[df_r.duplicated(subset=['Field_Id'], keep=False)]['Field_Id'].unique()

        for field_id in duplicated_field_ids:
            predictions_for_field = df_r[df_r['Field_Id'] == field_id]['test_predictions']
            if not predictions_for_field.empty:
                class_counts = predictions_for_field.value_counts(normalize=True) * 100
                df = pd.DataFrame(class_counts).T
                df['Field_Id'] = field_id
                df['test_predictions'] = predictions_for_field.iloc[0]
                dfs.append(df)

        df_percentage = pd.concat(dfs, ignore_index=True)
        df_percentage = df_percentage[['Field_Id', 'test_predictions', 6, 8, 9]]
        return df_percentage


    def create_final_dataframe(self, df_percentage, test_df_shuffled):
        '''
        Creates the final DataFrame with final predictions.
        Input:
            df_percentage (DataFrame): DataFrame with prediction percentages
            test_df_shuffled (DataFrame): Shuffled test data
        Output:
            df_final (DataFrame): Final DataFrame with predictions
        '''
        df_final = pd.DataFrame(columns=['Field_Id', 'Crop_Id_Ne', 'test_predictions'])

        for field_id in df_percentage['Field_Id'].unique():
            row = df_percentage[df_percentage['Field_Id'] == field_id].iloc[0]

            if row[6] > 70 or row[8] > 70:
                max_class = max(row[[6, 8]].idxmax(), 6)
            else:
                max_class = 9
            crop_id_ne = test_df_shuffled[test_df_shuffled['Field_Id'] == field_id]['Crop_Id_Ne'].iloc[0]

            df_final = pd.concat([df_final, pd.DataFrame({'Field_Id': [field_id], 'Crop_Id_Ne': [crop_id_ne], 'test_predictions': [max_class]})], ignore_index=True)
        return df_final






class ModelTrainer:
    def __init__(self, rf_params):
        self.rf_params = rf_params
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.class_weight = None  

    @staticmethod
    def calculate_class_weights(df):
        '''
        Calculates class weights to handle data imbalance.
        Input:
            df (DataFrame): DataFrame containing the data
        Output:
            class_weights (dict): Dictionary of class weights
        '''
        class_counts = df['Crop_Id_Ne'].value_counts()
        total_samples = len(df)
        num_classes = len(class_counts)
        class_weights = {}
        for class_label, count in class_counts.items():
            weight = total_samples / (count * num_classes)
            class_weights[class_label] = weight
        return class_weights

    def prepare_train_test_data(self, train_df, test_df):
        '''
        Prepares training and test data.
        Input:
            train_df (DataFrame): Training data
            test_df (DataFrame): Test data
        '''
        self.X_train = train_df.drop(['Crop_Id_Ne', 'Crop_Type', 'Pixel_Id', 'Field_Id'], axis=1)
        self.y_train = train_df['Crop_Id_Ne']
        self.X_test = test_df.drop(['Crop_Id_Ne', 'Crop_Type', 'Pixel_Id', 'Field_Id'], axis=1)
        self.y_test = test_df['Crop_Id_Ne']

    def train_model(self):
        '''
        Trains the RandomForest model.
        '''
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data is not provided. Call prepare_train_test_data first.")
        
        rf_params_with_weight = self.rf_params.copy() 
        rf_params_with_weight['class_weight'] = self.class_weight  
        self.rf_classifier = RandomForestClassifier(**rf_params_with_weight)
        self.rf_classifier.fit(self.X_train, self.y_train)

    def save_model(self, model_path):
        '''
        Saves the trained model.
        Input:
            model_path (str): Path to save the model
        '''
        dump(self.rf_classifier, model_path)
        
    def save_model_params(self):
        '''
        Returns the model parameters.
        Output:
            dict: Model parameters
        '''
        return self.rf_classifier.get_params()       

    def evaluate_model(self):
        '''
        Evaluates the model performance.
        Output:
            None (prints evaluation metrics)
        '''
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data is not provided. Call prepare_train_test_data first.")
        train_predictions = self.rf_classifier.predict(self.X_train)
        test_predictions = self.rf_classifier.predict(self.X_test)
        pred_prob = self.rf_classifier.predict_proba(self.X_test)
        
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("Classification Report for Test Data:")
        print(classification_report(self.y_test, test_predictions))




class PredictionConverter:
    def __init__(self, test_df, test_predictions):
        self.df_r = pd.DataFrame({'Field_Id': test_df['Field_Id']})
        self.df_r['test_predictions'] = test_predictions
        self.classes = self.df_r['test_predictions'].unique().tolist()

    def calculate_predictions_percentage(self):
        '''
        Calculates prediction percentages for each Field_Id.
        Output:
            df_percentage (DataFrame): DataFrame with prediction percentages
        '''

        dfs = []
        duplicated_field_ids = self.df_r[self.df_r.duplicated(subset=['Field_Id'], keep=False)]['Field_Id'].unique()
        self.classes = self.df_r['test_predictions'].unique().tolist()

        for field_id in duplicated_field_ids:
            predictions_for_field = self.df_r[self.df_r['Field_Id'] == field_id]['test_predictions']
            if not predictions_for_field.empty:
                class_counts = predictions_for_field.value_counts(normalize=True) * 100
                df_temp = pd.DataFrame(class_counts).T
                df_temp['Field_Id'] = field_id
                df_temp['test_predictions'] = predictions_for_field.iloc[0]
                dfs.append(df_temp)

        df_percentage = pd.concat(dfs, ignore_index=True)
        df_percentage = df_percentage[['Field_Id', 'test_predictions'] + self.classes]
        return df_percentage

    def calculate_final_predictions(self, test_df, df_percentage):
        '''
        Calculates final predictions and accuracy.
        Input:
            test_df (DataFrame): Test data
            df_percentage (DataFrame): DataFrame with prediction percentages
        Output:
            df_final (DataFrame): DataFrame with final predictions
        '''
        df_final = pd.DataFrame(columns=['Field_Id', 'Crop_Id_Ne', 'test_predictions'])
        for field_id in df_percentage['Field_Id'].unique():
            row = df_percentage[df_percentage['Field_Id'] == field_id].iloc[0]
            max_class = row.iloc[2:].idxmax()
            crop_id_ne = test_df[test_df['Field_Id'] == field_id]['Crop_Id_Ne'].iloc[0]
            df_final = pd.concat([df_final, pd.DataFrame({'Field_Id': [field_id], 'Crop_Id_Ne': [crop_id_ne], 'test_predictions': [max_class]})], ignore_index=True)
        accuracy = (df_final['Crop_Id_Ne'] == df_final['test_predictions']).mean() * 100
        print("Accuracy  {:.2f}%".format(accuracy))
            
        return df_final