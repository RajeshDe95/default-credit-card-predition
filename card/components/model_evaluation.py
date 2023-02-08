from card.entity import artifact_entity,config_entity
from card.exception import CardException
from card.logger import logging
from card.predictor import ModelResolver
import os,sys
from card.utils import load_object
from card.config import TARGET_COLUMN
import pandas as pd
#from card.utils import convert_to_yes_no
from sklearn.metrics import  accuracy_score


class ModelEvaluation:

    def __init__(self,model_eval_config:config_entity.ModelEvaluationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                    model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver()
        except Exception as e:
            raise CardException(e, sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
                improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            #finding location of transformer,model from previous trained model
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()


            logging.info("Previous trained objects of transformer, model")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)

            logging.info("Currently trained model objects")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model  = load_object(file_path=self.model_trainer_artifact.model_path)


            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            #y_true = test_df[TARGET_COLUMN].map({1: 'yes', 0: 'no'})


            # accuracy using previous trained model
            input_feature_name = list(transformer.feature_names_in_)
            input_arr =transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model: {y_pred[:5]}")
            previous_model_score = accuracy_score(y_true=target_df, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")


            # accuracy using current trained model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr =current_transformer.transform(test_df[input_feature_name])
            y_pred = current_model.predict(input_arr)
            #y_true = test_df[TARGET_COLUMN].map({1: 'yes', 0: 'no'})
            print(f"Prediction using trained model: {y_pred[:5]}")
            current_model_score = accuracy_score(y_true=target_df, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")

            #checking which model is better
            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")


            #prepare for artifact
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact


        except Exception as e:
            raise CardException(e, sys)


