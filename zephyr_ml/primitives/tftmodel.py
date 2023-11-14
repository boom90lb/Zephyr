import torch
import torch.utils.data
import pytorch_forecasting as pytf
import lightning.pytorch as lightning
from pytorch_forecasting.metrics import CrossEntropy
import pandas as pd
import numpy as np
import csv

def get_column_names(data_path="."):
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            return row
        
def train(training_data, validation_data):
    model = pytf.TemporalFusionTransformer.from_dataset(training_data, **TFT_PARAMS)
    # training_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True)
    # validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=128, shuffle=True)
    training_loader = training_data.to_dataloader(train=True, batch_size=8)
    validation_loader = validation_data.to_dataloader(train=False, batch_size=16)

    trainer = lightning.Trainer(**TRAINER_PARAMS)
    trainer.fit(model, training_loader, validation_loader)

    return trainer

TRAIN_PCT = 0.7
VAL_PCT = 0.85
DATA_PATH = "/mnt/zephyr/modeling/tft_format.csv"
COLUMN_NAMES = ['turbine_id', 'time', 'label', 'timestamp', 'turbine', 'WGEN_IntTmp_max', 'WGEN_IntTmp_mean', 'WGEN_IntTmp_min', 'WGEN_IntTmp_sd', 'WGEN_ClSt_max', 'WGEN_ClSt_mean', 'WGEN_ClSt_min', 'WCNV_Torq_mean', 'WCNV_Torq_max', 'WGEN_Spd_max', 'WGEN_Spd_mean', 'WGEN_Spd_min', 'WGEN_W_mean', 'WGEN_W_max', 'WCNV_Torq_sd', 'WGEN_Spd_sd', 'WGEN_W_min', 'WGEN_W_sd', 'WTOW_PlatTmp_mean', 'WNAC_WdSpd1_mean', 'WNAC_WdSpd1_max', 'WNAC_WdSpd2_max', 'WNAC_WdSpd2_mean', 'WNAC_WdDir1_mean', 'WNAC_WdDir2_mean', 'WNAC_ExtPres_mean', 'WNAC_WdSpdAvg_mean', 'WNAC_WdSpdAvg_max', 'WNAC_WdSpdAvg_min', 'WNAC_IntTmp_mean', 'WNAC_IntTmp_max', 'WNAC_IntTmp_min', 'WNAC_IntTmp_sd', 'WNAC_Vib1_max', 'WNAC_Vib1_min', 'WNAC_Vib1_mean', 'WNAC_Vib2_max', 'WNAC_Vib2_mean', 'WNAC_Vib2_min', 'WNAC_Vib3_max', 'WNAC_Vib3_mean', 'WNAC_Vib3_min', 'WNAC_Vib4_max', 'WNAC_Vib4_mean', 'WNAC_Vib4_min', 'WTRM_FtrPres1_mean', 'WTRM_FtrPres1_max', 'WTRM_FtrPres2_max', 'WTRM_FtrPres2_mean', 'WTRM_FtrPres1_min', 'WTRM_FtrPres2_min', 'WROT_Spd1_mean', 'WROT_Spd1_max', 'WROT_Spd1_min', 'WROT_Spd1_sd', 'WROT_Spd2_max', 'WROT_Spd2_mean', 'WROT_Spd2_min', 'WROT_Spd2_sd', 'WROT_Pos_max', 'WROT_Pos_mean', 'WROT_Pos_min', 'WROT_MnBrgTemp1_max', 'WROT_MnBrgTemp1_mean', 'WROT_MnBrgTemp1_min', 'WROT_MnBrgTemp2_max', 'WROT_MnBrgTemp2_mean', 'WROT_MnBrgTemp2_min', 'WTRM_HyFtrPres1_mean', 'WTRM_HyFtrPres1_max', 'WTRM_HyFtrPres1_min', 'WTRM_HyFtrPres1_sd', 'WTRM_HyFtrPres2_max', 'WTRM_HyFtrPres2_mean', 'WTRM_HyFtrPres2_min', 'WTRM_HyFtrPres2_sd', 'WTRM_HySysPres1_max', 'WTRM_HySysPres1_mean', 'WTRM_HySysPres1_min', 'WTRM_HySysLockPres1_max', 'WTRM_HySysLockPres1_mean', 'WTRM_HySysLockPres1_sd', 'WTRM_HySysLockPres1_min', 'WROT_LockPos1_mean', 'WROT_LockPos1_max', 'WROT_LockPos1_min', 'WROT_LockPos1_sd', 'WROT_LockPos2_max', 'WROT_LockPos2_mean', 'WROT_LockPos2_min', 'WROT_LockPos2_sd', 'WROT_LockPos3_max', 'WROT_LockPos3_mean', 'WROT_LockPos3_min', 'WROT_LockPos3_sd', 'WROT_Brk2HyTmp6_sd', 'WROT_Brk2HyTmp6_min', 'WROT_Brk2HyTmp6_mean', 'WROT_Brk2HyTmp6_max', 'WROT_Brk2HyTmp5_sd', 'WROT_Brk2HyTmp5_min', 'WROT_Brk2HyTmp5_mean', 'WROT_Brk2HyTmp5_max', 'WROT_Brk1HyTmp6_min', 'WROT_Brk1HyTmp6_sd', 'WROT_Brk1HyTmp6_mean', 'WROT_Brk1HyTmp6_max', 'WROT_Brk1HyTmp5_sd', 'WROT_Brk1HyTmp5_min', 'WROT_Brk1HyTmp5_mean', 'WROT_Brk1HyTmp5_max', 'WROT_Brk2HyTmp4_sd', 'WROT_Brk2HyTmp4_min', 'WROT_Brk2HyTmp4_max', 'WROT_Brk2HyTmp3_sd', 'WROT_Brk2HyTmp4_mean', 'WROT_Brk2HyTmp3_min', 'WROT_Brk2HyTmp3_mean', 'WROT_Brk2HyTmp3_max', 'WROT_Brk1HyTmp4_sd', 'WROT_Brk1HyTmp4_min', 'WROT_Brk1HyTmp4_mean', 'WROT_Brk1HyTmp4_max', 'WROT_Brk1HyTmp3_sd', 'WROT_Brk1HyTmp3_min', 'WROT_Brk1HyTmp3_mean', 'WROT_Brk1HyTmp3_max', 'WROT_Brk2HyTmp2_sd', 'WROT_Brk2HyTmp2_min', 'WROT_Brk2HyTmp2_mean', 'WROT_Brk2HyTmp2_max', 'WROT_HyOilTmp1_sd', 'WROT_HyOilTmp1_min', 'WROT_HyOilTmp1_mean', 'WROT_HyOilTmp1_max', 'WROT_Brk2HyTmp1_sd', 'WROT_Brk2HyTmp1_min', 'WROT_Brk2HyTmp1_mean', 'WROT_Brk2HyTmp1_max', 'WROT_Brk1HyTmp2_sd', 'WROT_Brk1HyTmp2_min', 'WROT_Brk1HyTmp2_mean', 'WROT_Brk1HyTmp2_max', 'WROT_Brk1HyTmp1_sd', 'WROT_Brk1HyTmp1_min', 'WROT_Brk1HyTmp1_mean', 'WROT_Brk1HyTmp1_max', 'WROT_Brk1HyPres_max', 'WROT_Brk1HyPres_mean', 'WROT_Brk1HyPres_min', 'WROT_Brk1HyPres_sd', 'WROT_Brk2HyPres_max', 'WROT_Brk2HyPres_mean', 'WROT_Brk2HyPres_min', 'WROT_Brk2HyPres_sd', 'WROT_Brk1HyAccPres_max', 'WROT_Brk1HyAccPres_mean', 'WROT_Brk1HyAccPres_min', 'WROT_Brk1HyAccPres_sd', 'WROT_Brk2HyAccPres_max', 'WROT_Brk2HyAccPres_mean', 'WROT_Brk2HyAccPres_min', 'WROT_Brk2HyAccPres_sd']
if COLUMN_NAMES == None:
    COLUMN_NAMES = get_column_names(DATA_PATH)
# Define parameters for TimeSeriesDataSet
TSDS_PARAMS = {
    "time_idx": "time_idx",
    "target": "label",
    "group_ids": [COLUMN_NAMES[4]],
    "min_encoder_length": 1,
    "max_encoder_length": 64,
    "min_prediction_length": 1,
    "max_prediction_length": 32,
    "static_categoricals": [],
    "static_reals": [],
    "time_varying_known_reals": [],
    "time_varying_unknown_reals": COLUMN_NAMES[5:-1] + ["label"],
    "time_varying_unknown_categoricals": [],
    "allow_missing_timesteps": True,
}
# Define parameters for TemporalFusionTransformer
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
# Define parameters for Trainer
TRAINER_PARAMS = {
    "max_epochs": 10,
    "accelerator": "auto",
}

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by='time').reset_index(drop=True)
    df['time_idx'] = range(len(df))
    df['label'] = df['label'].astype(float)

    training_cutoff = np.floor(len(df)*TRAIN_PCT).item()
    validation_cutoff = np.floor(len(df)*VAL_PCT).item()

    training_data = pytf.TimeSeriesDataSet(df.loc[df['time_idx'] <= training_cutoff], **TSDS_PARAMS)

    validation_data = pytf.TimeSeriesDataSet(df.loc[df['time_idx'] <= validation_cutoff], **TSDS_PARAMS, min_prediction_idx=training_cutoff+1)

    test_data = pytf.TimeSeriesDataSet(df, **TSDS_PARAMS, min_prediction_idx=validation_cutoff+1)

    trained_model = train(training_data, validation_data)