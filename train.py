from card.pipeline.training_pipeline import start_training_pipeline



file_path="/config/workspace/EDA_data_of_Credit_Card.csv"
print(__name__)
if __name__=="__main__":
     try:
        start_training_pipeline()

     except  Exception as e:
          print(e) 