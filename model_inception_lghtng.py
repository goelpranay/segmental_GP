import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.optim.lr_scheduler import OneCycleLR
class Inception(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.aux_logits = False
        num_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        logits = self.inception(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.unsqueeze(1)
        loss = F.mse_loss(logits, y)
        self.log('train_loss', loss)
        # return loss
        tensorboard_logs = {'t_train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.unsqueeze(1)
        loss = F.l1_loss(logits, y)
        self.log('val_loss', loss)
        # return loss

        tensorboard_logs = {'t_val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

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

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=50, verbose=True, threshold=0.001
        # )
        scheduler = OneCycleLR(optimizer, max_lr=1e-2, total_steps=1000)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self(x)
