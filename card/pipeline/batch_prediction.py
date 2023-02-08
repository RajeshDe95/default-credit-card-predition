from card.entity import artifact_entity,config_entity
from card.exception import CardException
from card.logger import logging
from card.predictor import ModelResolver
import os,sys
import pandas as pd
import numpy as np
from datetime import datetime
from card.utils import load_object
PREDICTION_DIR="prediction"


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file:{input_file_path}")
        df = pd.read_csv(input_file_path)

        logging.info(f"loading transformer to transform the dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

        input_feature_names =  list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)


        #cat_prediction = prediction.map({1: 'yes', 0: 'no'})
        cat_prediction = np.vectorize({1: 'yes', 0: 'no'}.get)(prediction)

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path

    except Exception as e:
        raise CardException(e, sys)


