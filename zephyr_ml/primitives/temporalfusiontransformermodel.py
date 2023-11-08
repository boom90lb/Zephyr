import pytorch_forecasting as pytf

## Produce will use the predict method of Trainer rather than forward of TFT, Produce here only returns an initialized model

class TemporalFusionTransformer:
    def __init__(self, hidden_size=256, lstm_layers=2) -> None:
        self.model = pytf.TemporalFusionTransformer(hidden_size=256)

    def produce(self, X):
        self.model = self.model.from_dataset(X, hidden_size=256)