import torch
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import inception_v3, Inception_V3_Weights

def squared_epsilon_insensitive_loss(y_hat, y, epsilon):
    """
    Computes the squared epsilon-insensitive loss between y_hat and y.

    Args:
        y_hat (torch.Tensor): Predicted target values, of shape (batch_size,).
        y (torch.Tensor): True target values, of shape (batch_size,).
        epsilon (float): Size of the insensitive region.

    Returns:
        torch.Tensor: The squared epsilon-insensitive loss, of shape ().
    """
    loss = torch.max(torch.tensor(0, dtype=torch.float32), torch.abs(y - y_hat) - epsilon) ** 2
    return loss.mean()

class ResNet50(pl.LightningModule):
    def __init__(self, epsilon=6):
        super().__init__()

        self.epsilon = epsilon

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, 1)
        # self.resnet = resnet152(weights=ResNet152_Weights.DEFAULT) # which resnet to use?
        # num_features = self.resnet.fc.in_features
        # self.resnet.fc = torch.nn.Linear(num_features, 1)


    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = squared_epsilon_insensitive_loss(logits, y, self.epsilon)
        # loss = torch.nn.functional.mse_loss(logits, y.float().unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.l1_loss(logits, y.float().unsqueeze(1))
        self.log("val_loss", loss)
        return {"val_loss": loss, "val_preds": logits}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = torch.cat([x["val_preds"] for x in outputs], dim=0)
        self.logger.experiment.add_scalar(
            "val_loss_epoch", val_loss_mean, self.current_epoch
        )
        self.log("val_loss", val_loss_mean, prog_bar=True)
        return {"val_loss": val_loss_mean, "val_preds": preds}

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        fc_list = ["fc.weight", "fc.bias"]
        fc_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] in fc_list, self.resnet.named_parameters())
                ),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(
                        lambda kv: kv[0] not in fc_list, self.resnet.named_parameters()
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)
