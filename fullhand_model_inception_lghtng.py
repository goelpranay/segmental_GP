import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import inception_v3, Inception_V3_Weights

class Inception(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT) # hack to use inception instead
        num_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        logits = self.inception(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets_resized = y.view(logits.size())
        loss = F.mse_loss(logits, targets_resized)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets_resized = y.view(logits.size())
        loss = F.mse_loss(logits, targets_resized)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outputs):
        # val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        # preds = torch.cat([x["val_preds"] for x in outputs], dim=0)
        # self.logger.experiment.add_scalar(
        #     "val_loss_epoch", val_loss_mean, self.current_epoch
        # )
        # self.log("val_loss", val_loss_mean, prog_bar=True)
        # return {"val_loss": val_loss_mean, "val_preds": preds}

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss,prog_bar=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        fc_list = ["fc.weight", "fc.bias"]
        fc_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] in fc_list, self.inception.named_parameters())
                ),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(
                        lambda kv: kv[0] not in fc_list, self.inception.named_parameters()
                    )
                ),
            )
        )
        optimizer = torch.optim.Adam(
            [
                {"params": base_params},
                {"params": fc_params, "lr": 5e-3},
            ],
            lr=5e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True, threshold=0.001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self(x)
