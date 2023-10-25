import torch
import pytorch_forecasting as pytf
import pytorch_lightning as lightning
import pandas as pd
import csv

data_paths = ["/mnt/zephyr/modeling/target_times.csv", "/mnt/zephyr/modeling/timeseries.csv", "/mnt/zephyr/modeling/metadata.csv"]
# column_names[0] are target_times labels, column_names[1] are timeseries labels, column_names[2] are metadata labels, this is for reference/use if needed
column_names = [['turbine_id', 'time', 'label'],
                ['timestamp', 'turbine', 'WGEN_IntTmp_max', 'WGEN_IntTmp_mean', 'WGEN_IntTmp_min', 'WGEN_IntTmp_sd', 'WGEN_ClSt_max', 'WGEN_ClSt_mean', 'WGEN_ClSt_min', 'WCNV_Torq_mean', 'WCNV_Torq_max', 'WGEN_Spd_max', 'WGEN_Spd_mean', 'WGEN_Spd_min', 'WGEN_W_mean', 'WGEN_W_max', 'WCNV_Torq_sd', 'WGEN_Spd_sd', 'WGEN_W_min', 'WGEN_W_sd', 'WTOW_PlatTmp_mean', 'WNAC_WdSpd1_mean', 'WNAC_WdSpd1_max', 'WNAC_WdSpd2_max', 'WNAC_WdSpd2_mean', 'WNAC_WdDir1_mean', 'WNAC_WdDir2_mean', 'WNAC_ExtPres_mean', 'WNAC_WdSpdAvg_mean', 'WNAC_WdSpdAvg_max', 'WNAC_WdSpdAvg_min', 'WNAC_IntTmp_mean', 'WNAC_IntTmp_max', 'WNAC_IntTmp_min', 'WNAC_IntTmp_sd', 'WNAC_Vib1_max', 'WNAC_Vib1_min', 'WNAC_Vib1_mean', 'WNAC_Vib2_max', 'WNAC_Vib2_mean', 'WNAC_Vib2_min', 'WNAC_Vib3_max', 'WNAC_Vib3_mean', 'WNAC_Vib3_min', 'WNAC_Vib4_max', 'WNAC_Vib4_mean', 'WNAC_Vib4_min', 'WTRM_FtrPres1_mean', 'WTRM_FtrPres1_max', 'WTRM_FtrPres2_max', 'WTRM_FtrPres2_mean', 'WTRM_FtrPres1_min', 'WTRM_FtrPres2_min', 'WROT_Spd1_mean', 'WROT_Spd1_max', 'WROT_Spd1_min', 'WROT_Spd1_sd', 'WROT_Spd2_max', 'WROT_Spd2_mean', 'WROT_Spd2_min', 'WROT_Spd2_sd', 'WROT_Pos_max', 'WROT_Pos_mean', 'WROT_Pos_min', 'WROT_MnBrgTemp1_max', 'WROT_MnBrgTemp1_mean', 'WROT_MnBrgTemp1_min', 'WROT_MnBrgTemp2_max', 'WROT_MnBrgTemp2_mean', 'WROT_MnBrgTemp2_min', 'WTRM_HyFtrPres1_mean', 'WTRM_HyFtrPres1_max', 'WTRM_HyFtrPres1_min', 'WTRM_HyFtrPres1_sd', 'WTRM_HyFtrPres2_max', 'WTRM_HyFtrPres2_mean', 'WTRM_HyFtrPres2_min', 'WTRM_HyFtrPres2_sd', 'WTRM_HySysPres1_max', 'WTRM_HySysPres1_mean', 'WTRM_HySysPres1_min', 'WTRM_HySysLockPres1_max', 'WTRM_HySysLockPres1_mean', 'WTRM_HySysLockPres1_sd', 'WTRM_HySysLockPres1_min', 'WROT_LockPos1_mean', 'WROT_LockPos1_max', 'WROT_LockPos1_min', 'WROT_LockPos1_sd', 'WROT_LockPos2_max', 'WROT_LockPos2_mean', 'WROT_LockPos2_min', 'WROT_LockPos2_sd', 'WROT_LockPos3_max', 'WROT_LockPos3_mean', 'WROT_LockPos3_min', 'WROT_LockPos3_sd', 'WROT_Brk2HyTmp6_sd', 'WROT_Brk2HyTmp6_min', 'WROT_Brk2HyTmp6_mean', 'WROT_Brk2HyTmp6_max', 'WROT_Brk2HyTmp5_sd', 'WROT_Brk2HyTmp5_min', 'WROT_Brk2HyTmp5_mean', 'WROT_Brk2HyTmp5_max', 'WROT_Brk1HyTmp6_min', 'WROT_Brk1HyTmp6_sd', 'WROT_Brk1HyTmp6_mean', 'WROT_Brk1HyTmp6_max', 'WROT_Brk1HyTmp5_sd', 'WROT_Brk1HyTmp5_min', 'WROT_Brk1HyTmp5_mean', 'WROT_Brk1HyTmp5_max', 'WROT_Brk2HyTmp4_sd', 'WROT_Brk2HyTmp4_min', 'WROT_Brk2HyTmp4_max', 'WROT_Brk2HyTmp3_sd', 'WROT_Brk2HyTmp4_mean', 'WROT_Brk2HyTmp3_min', 'WROT_Brk2HyTmp3_mean', 'WROT_Brk2HyTmp3_max', 'WROT_Brk1HyTmp4_sd', 'WROT_Brk1HyTmp4_min', 'WROT_Brk1HyTmp4_mean', 'WROT_Brk1HyTmp4_max', 'WROT_Brk1HyTmp3_sd', 'WROT_Brk1HyTmp3_min', 'WROT_Brk1HyTmp3_mean', 'WROT_Brk1HyTmp3_max', 'WROT_Brk2HyTmp2_sd', 'WROT_Brk2HyTmp2_min', 'WROT_Brk2HyTmp2_mean', 'WROT_Brk2HyTmp2_max', 'WROT_HyOilTmp1_sd', 'WROT_HyOilTmp1_min', 'WROT_HyOilTmp1_mean', 'WROT_HyOilTmp1_max', 'WROT_Brk2HyTmp1_sd', 'WROT_Brk2HyTmp1_min', 'WROT_Brk2HyTmp1_mean', 'WROT_Brk2HyTmp1_max', 'WROT_Brk1HyTmp2_sd', 'WROT_Brk1HyTmp2_min', 'WROT_Brk1HyTmp2_mean', 'WROT_Brk1HyTmp2_max', 'WROT_Brk1HyTmp1_sd', 'WROT_Brk1HyTmp1_min', 'WROT_Brk1HyTmp1_mean', 'WROT_Brk1HyTmp1_max', 'WROT_Brk1HyPres_max', 'WROT_Brk1HyPres_mean', 'WROT_Brk1HyPres_min', 'WROT_Brk1HyPres_sd', 'WROT_Brk2HyPres_max', 'WROT_Brk2HyPres_mean', 'WROT_Brk2HyPres_min', 'WROT_Brk2HyPres_sd', 'WROT_Brk1HyAccPres_max', 'WROT_Brk1HyAccPres_mean', 'WROT_Brk1HyAccPres_min', 'WROT_Brk1HyAccPres_sd', 'WROT_Brk2HyAccPres_max', 'WROT_Brk2HyAccPres_mean', 'WROT_Brk2HyAccPres_min', 'WROT_Brk2HyAccPres_sd', 'turbine_id'],
                ['turbine_id', 'DES_TECH_NAME', 'COD_STATUS_PAUSE', 'COD_STATUS_STOP', 'PREV_ORDERS', 'COD_ORDER_TYPE_RD', 'COD_ORDER_TYPE_RM', 'COD_ORDER_TYPE_RP']]


df = pd.concat((pd.read_csv(f) for f in (data_paths)), ignore_index=True)
for i in range(len(df)):
    # set time_idx based on timestamp (same date = same idx)
    pass
    

# Define parameters for TimeSeriesDataSet
TSDSParams = {
    "time_idx": "time",
    "target": "label",
    "group_ids": ["turbine_id"],
    "max_encoder_length": 256, 
    "min_prediction_idx": 1024, 
    "max_prediction_length": 128, 
    "static_categoricals": column_names[2], 
    "time_varying_known_reals": column_names[1][2:len(column_names[1])-1], 
    "time_varying_unknown_reals": "time", 
    "time_varying_unknown_categoricals": "label", 
    "allow_missing_timesteps": True,
}

data = pytf.TimeSeriesDataSet(df, **TSDSParams)

# Define parameters for TemporalFusionTransformer
TFTparams = {
    "hidden_size": 16,
    "lstm_layers": 2,
    "dropout": 0.2,
    "output_size": 1, 
    "attention_head_size": 4,
    "max_encoder_length": 256,
    "static_categoricals": column_names[2],
    "static_reals": [], 
    "time_varying_categoricals_encoder": [], 
    "time_varying_categoricals_decoder": ["label"],
    "time_varying_reals_encoder": column_names[1][2:len(column_names[1])-1],
    "time_varying_reals_decoder": ["time"],
    "categorical_groups": {}, 
    "x_reals": column_names[1][2:len(column_names[1])-1],
    "x_categoricals": ["label"],
    "hidden_continuous_size": 8, 
    "learning_rate": 0.001, 
    "log_interval": 10,
    "log_val_interval": 1,  
    "reduce_on_plateau_patience": 4,
    "monotone_constraints": {},
    "share_single_variable_networks": False,
    "causal_attention": True,
}
model = pytf.TemporalFusionTransformer.from_dataset(data, **TFTparams)

# Define parameters for Trainer
trainer_params = {
    "max_epochs": 10,
    "accelerator": "auto",
}

trainer = lightning.Trainer(**trainer_params)
trainer.fit(model, data)



# with open(data_paths[2]) as path:
#     csv.reader(path)
#     i=0
#     for row in csv.reader(path):
#         i = i+1
#         print(row)
#         if i > 10:
#             break


## Make pipeline manually first