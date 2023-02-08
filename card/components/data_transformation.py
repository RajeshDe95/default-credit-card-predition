from card.entity import artifact_entity,config_entity
from card.exception import CardException
from card.logger import logging
import os,sys
import pandas as pd
from card import utils
import numpy as np
from imblearn.combine import SMOTETomek
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from card.config import TARGET_COLUMN




class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise CardException(e, sys)



    @classmethod
    def get_data_trnsformer_object(cls):
        try:
            robust_scaler =  RobustScaler()
            return robust_scaler
        except Exception as e:
            raise CardException(e, sys)




    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:
        try:
            #reading traing and testing file
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #selecting input feature for train and test dataset
            input_train_feature_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_test_feature_df=test_df.drop(TARGET_COLUMN,axis=1)

            #selecting Target feature for train and test dataset
            target_train_feature_df=train_df[TARGET_COLUMN]
            target_test_feature_df=test_df[TARGET_COLUMN]

            
            transformation_obj=DataTransformation.get_data_trnsformer_object()
            transformation_obj.fit(input_train_feature_df)

            #transforming input feature
            input_feature_train_arr = transformation_obj.transform(input_train_feature_df)
            input_feature_test_arr = transformation_obj.transform(input_test_feature_df)


            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_train_feature_df.shape}")
            input_feature_train_arr, target_train_feature_df = smt.fit_resample(input_feature_train_arr, target_train_feature_df)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_train_feature_df.shape}")


            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_test_feature_df.shape}")
            input_feature_test_arr, target_test_feature_df = smt.fit_resample(input_feature_test_arr, target_test_feature_df)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_test_feature_df.shape}")


            #concatinate both train and test 
            train_arr = np.c_[input_feature_train_arr, target_train_feature_df]
            test_arr = np.c_[input_feature_test_arr, target_test_feature_df]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_obj)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CardException(e, sys)






    