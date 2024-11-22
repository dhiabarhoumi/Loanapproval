import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        person_gender: str,
        person_education: str,
        person_income: float,
        person_age: float,
        person_emp_exp: int,
        person_home_ownership: str,
        loan_amnt: float,
        loan_intent: str,
        loan_int_rate: float,
        loan_percent_income: float,
        cb_person_cred_hist_length: float,
        credit_score: int,
        previous_loan_defaults_on_file: str
        ):

        self.person_gender = person_gender

        self.person_education = person_education

        self.person_income = person_income

        self.person_age = person_age

        self.person_emp_exp = person_emp_exp

        self.person_home_ownership = person_home_ownership

        self.loan_amnt = loan_amnt

        self.loan_intent= loan_intent

        self.loan_int_rate= loan_int_rate

        self.loan_percent_income=loan_percent_income/100

        self.cb_person_cred_hist_length= cb_person_cred_hist_length

        self.credit_score=credit_score

        self.previous_loan_defaults_on_file=previous_loan_defaults_on_file



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "person_gender": [self.person_gender],
                "person_education": [self.person_education],
                "person_income": [self.person_income],
                "person_age": [self.person_age],
                "person_emp_exp": [self.person_emp_exp],
                "person_home_ownership": [self.person_home_ownership],
                "loan_amnt": [self.loan_amnt],
                "loan_intent": [self.loan_intent],
                "loan_int_rate": [self.loan_int_rate],
                "loan_percent_income": [self.loan_percent_income],
                "cb_person_cred_hist_length": [self.cb_person_cred_hist_length],
                "credit_score": [self.credit_score],
                "previous_loan_defaults_on_file": [self.previous_loan_defaults_on_file]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)