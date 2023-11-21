import pytorch_forecasting as pytf
import lightning.pytorch as lightning
from pytorch_forecasting.metrics import CrossEntropy
import pandas as pd
import numpy as np
import csv
from pytorch_forecasting.data.examples import get_stallion_data

data = get_stallion_data()

def get_column_names(data_path="."):
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            return row

def train(training_data, validation_data):
    model = pytf.TemporalFusionTransformer.from_dataset(training_data, **TFT_PARAMS)
    
    training_loader = training_data.to_dataloader(train=True, batch_size=8)
    validation_loader = validation_data.to_dataloader(train=False, batch_size=16)

    trainer = lightning.Trainer(**TRAINER_PARAMS)
    trainer.fit(model, training_loader, validation_loader)

    return trainer

TRAIN_PCT = 0.7
VAL_PCT = 0.85
DATA_PATH = "/mnt/zephyr/modeling/tft_format.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)
# Ensure 'turbine' is a categorical column, turn 'label' from bool to float
df['label'] = df['label'].astype(float)
df['turbine'] = df['turbine'].astype('category')
# Extract column names sans 'time_idx' from the dataframe
COLUMN_NAMES = df.columns.tolist()
# Add time_idx
df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
df = df.sort_values(by='time').reset_index(drop=True)
df['time_idx'] = range(len(df))

training_cutoff = np.floor(len(df)*TRAIN_PCT).item()
validation_cutoff = np.floor(len(df)*VAL_PCT).item()

TSDS_PARAMS = {
    "time_idx": "time_idx",
    "target": "label",
    "group_ids": ["turbine"],
    "min_encoder_length": 1,
    "max_encoder_length": 64,
    "min_prediction_length": 1,
    "max_prediction_length": 8,
    "static_categoricals": [],
    "static_reals": [],
    "time_varying_known_reals": [],
    "time_varying_unknown_reals": COLUMN_NAMES[5:-1] + ["label"],
    "time_varying_unknown_categoricals": [],
    "allow_missing_timesteps": True,
}

TFT_PARAMS = {
    "hidden_size": 16,
    "lstm_layers": 2,
    "dropout": 0.2,
    "output_size": 1,
    "loss": CrossEntropy(),
    "attention_head_size": 4,
    "max_encoder_length": 128,
    "hidden_continuous_size": 8, 
    "learning_rate": 0.005, 
    "log_interval": 10,
    "optimizer": "Ranger",
    "log_val_interval": 1,  
    "reduce_on_plateau_patience": 4,
    "monotone_constaints": {},
    "share_single_variable_networks": False,
    "causal_attention": True,
}

TRAINER_PARAMS = {
    "max_epochs": 10,
    "accelerator": "auto",
}

if __name__ == "__main__":
    training_data = pytf.TimeSeriesDataSet(df.loc[df['time_idx'] <= training_cutoff], **TSDS_PARAMS)

    validation_data = pytf.TimeSeriesDataSet(df.loc[df['time_idx'] <= validation_cutoff], **TSDS_PARAMS, min_prediction_idx=training_cutoff+1)

    test_data = pytf.TimeSeriesDataSet(df, **TSDS_PARAMS, min_prediction_idx=validation_cutoff+1)

    trained_model = train(training_data, validation_data)
