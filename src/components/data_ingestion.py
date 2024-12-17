import os
import sys
from pathlib import Path

# Fix path issue
sys.path.append(str(Path(__file__).parent.parent))  # Correctly set project root

from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Define DataIngestionConfig class to hold file paths
@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(os.getcwd(), "artifacts")  # Absolute path for artifacts
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "data.csv")

# Define the DataIngestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initialize config

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Define the path to your dataset
            dataset_path = r"C:\mlproject\notebook\data\StudentsPerformance.csv"

            # Verify if dataset exists
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"File not found: {dataset_path}")
            
            # Read the dataset
            df = pd.read_csv(dataset_path)
            logging.info("Read the dataset into a dataframe")

            # Debugging: print dataset info
            print(f"Dataset Shape: {df.shape}")
            print(f"First 5 rows:\n{df.head()}")

            # Create artifacts directory
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)
            print(f"Artifacts Directory Created: {self.ingestion_config.artifacts_dir}")

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            print(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Perform Train-Test Split
            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            print(f"Train data saved at: {self.ingestion_config.train_data_path}")
            print(f"Test data saved at: {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

# Main block to run the script
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Train File Path: {train_path}")
    print(f"Test File Path: {test_path}")
