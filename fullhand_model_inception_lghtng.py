import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import inception_v3, Inception_V3_Weights

class Inception(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.aux_logits = False
        num_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(num_features, 1)

    # def sample_uniformly(self, y):
    #     labels = np.unique(y)
    #     label_weights = np.ones(len(labels))
    #     label_weights /= label_weights.sum()
    #     label_idx = np.random.choice(len(labels), p=label_weights)
    #     label = labels[label_idx]
    #     idx = np.where(y == label)[0]
    #     idx_sampled = np.random.choice(idx)
    #     return label, idx_sampled

    def forward(self, x):
        logits = self.inception(x)
        return logits

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     batch_size = len(y)
    #     labels = []
    #     idxs_sampled = []
    #     for i in range(batch_size):
    #         label, idx_sampled = self.sample_uniformly(y.numpy())
    #         labels.append(label)
    #         idxs_sampled.append(idx_sampled)
    #     x_sampled = x[idxs_sampled]
    #     y_sampled = torch.tensor(labels).unsqueeze(1)
    #     logits = self(x_sampled)
    #     loss = torch.nn.functional.cross_entropy(logits, y_sampled)
    #     self.log('train_loss', loss)
    #     # return loss
    #     tensorboard_logs = {'t_train_loss': loss}
    #     return {'loss': loss, 'log': tensorboard_logs}

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


    # def validation_epoch_end(self, outputs):
    #     # Assuming outputs is a list of dictionaries with a 'val_loss' key
    #     losses = [torch.tensor(x['val_loss'], requires_grad=False) for x in outputs]
    #
    #     # Calculate the mean loss over all batches
    #     avg_loss = torch.stack(losses).mean()
    #
    #     # Log the mean loss
    #     self.log('avg_val_loss', avg_loss)

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
