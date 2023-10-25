import torch
import pytorch_forecasting as pytf
import pytorch_lightning as lightning

## In init arguments, put in the important hyperparameters

## After initialization, include Trainer wrapper in the output of fit

## Predict will use the predict method of Trainer rather than forward of TFT

class TemporalFusionTransformer:
    def __init__(self, hidden_size=256, lstm_layers=2) -> None:
        self.model = pytf.TemporalFusionTransformer(hidden_size=256)

    def produce(self, X):
        self.model = self.model.from_dataset(X, hidden_size=256)