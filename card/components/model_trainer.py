from card.entity import artifact_entity,config_entity
from card.exception import CardException
from card.logger import logging
import os,sys 
from card import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
#from sklearn.metrics import f1_score
from sklearn.metrics import  accuracy_score


class ModetTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CardException(e, sys)



    def fine_tune(self,x,y):
        try:
            #Wite code for Grid Search CV
            grid_param = {'criterion': ['gini', 'entropy'],
                            'max_depth' : range(2,10,1),
                            'min_samples_leaf' : range(1,8,1),
                            'min_samples_split': range(2,8,1),
                            'oob_score':[True]}
            rfc = RandomForestClassifier()
            grid_search=GridSearchCV(estimator=rfc,param_grid=grid_param,cv=3,verbose=1)
            grid_search.fit(x,y)
            return grid_search
        except Exception as e:
            raise CardException(e, sys)


    def train_model(self,x,y):
        try:
            rf_clf = RandomForestClassifier()
            rf_clf.fit(x,y)
            return rf_clf
            #xgb_clf =  XGBClassifier()
            #xgb_clf.fit(x,y)
            #return xgb_clf
            #log_reg =  LogisticRegression()
            #log_reg.fit(x,y)
            #return log_reg

        except Exception as e:
            raise CardException(e, sys)



    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train)

            logging.info(f"Train the model with hyperparameter")
            model_tune = self.fine_tune(x=x_train,y=y_train)


            logging.info(f"Calculating train accuracy score")
            yhat_train = model_tune.predict(x_train)
            train_accuracy_score= accuracy_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating test accuracy score")
            yhat_test = model_tune.predict(x_test)
            test_accuracy_score = accuracy_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"train score:{train_accuracy_score} and tests score {test_accuracy_score}")

            #check for overfitting or underfiiting for expected score
            logging.info("Checking if our model is underfitting or not")
            if test_accuracy_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {test_accuracy_score}")

            logging.info("checking if the model is overfitting or not")
            diff = abs(train_accuracy_score-test_accuracy_score)
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            

            #save the trained model
            logging.info(f"Saving model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
            train_accuracy_score=train_accuracy_score, test_accuracy_score=test_accuracy_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise CardException(e, sys)







