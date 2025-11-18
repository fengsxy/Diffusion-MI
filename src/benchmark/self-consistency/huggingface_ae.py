import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from huggingface_hub import PyTorchModelHubMixin
import os

LATENT_DIM = 16
INPUT_DATA_DIM = 784
dataSize = torch.Size([1, 28, 28])
num_hidden_layers = 1

class MnistEncoder(nn.Module):
    def __init__(self, input_size=INPUT_DATA_DIM, latent_dim=LATENT_DIM, deterministic=False):
        super(MnistEncoder, self).__init__()
        self.deterministic = deterministic
        self.hidden_dim = 400

        modules = [nn.Sequential(nn.Linear(784, self.hidden_dim), nn.ReLU(True))]
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features=latent_dim, bias=True)
        self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features=latent_dim, bias=True)

    def forward(self, x):
        h = x.view(*x.size()[:-3], -1)
        h = self.enc(h)
        h = h.view(h.size(0), -1)

        latent_space_mu = self.hidden_mu(h)
        latent_space_logvar = self.hidden_logvar(h)
        if self.deterministic:
            return latent_space_mu
        else:
            return latent_space_mu, latent_space_logvar


class MnistDecoder(nn.Module):
    def __init__(self, input_size=INPUT_DATA_DIM, latent_dim=LATENT_DIM):
        super(MnistDecoder, self).__init__()
        self.hidden_dim = 400
        modules = [nn.Sequential(nn.Linear(latent_dim, self.hidden_dim), nn.ReLU(True))]
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x_hat = self.dec(z)
        x_hat = self.fc3(x_hat)
        x_hat = self.sigmoid(x_hat)
        x_hat = x_hat.view(*z.size()[:-1], *dataSize)
        return x_hat


class AE(pl.LightningModule, PyTorchModelHubMixin):

    def __init__(self, mod_name, test_loader=None, enc=None, dec=None, latent_dim=16, rows=28,
                 regularization=None, alpha=0.0, lr=0.001, decay=0.0, train_loader=None):
        super(AE, self).__init__()
        self.lr = lr
        self.decay = decay
        self.modality = mod_name
        self.latent_dim = latent_dim

        self.encoder = enc if enc is not None else MnistEncoder(latent_dim=latent_dim)
        self.decoder = dec if dec is not None else MnistDecoder(latent_dim=latent_dim)
        self.regularization = regularization
        self.test_loader = test_loader
        self.alpha = alpha
        self.rows = rows
        self.train_loader = train_loader
        self.save_hyperparameters(ignore=["modality", "encoder", "test_loader", "decoder", "crop_rate"])
        self.loss_func = nn.MSELoss(reduction="sum")

    def get_mod_cropped(self, x):
        x[:, :, self.rows:, :] = 0.0
        return x

    def training_step(self, x, batch_idx):
        x = self.get_mod_cropped(x[0])
        recon, z = self.forward(x)
        
        regularization = 0.0
        if self.regularization == "l1":
            regularization = torch.abs(z).sum()
        elif self.regularization == "l2":
            regularization = torch.square(z).sum()
                 
        recon_loss = self.reconstruction_loss(x, recon)
        total_loss = recon_loss + self.alpha * regularization

        self.log("train_loss", total_loss, prog_bar=True)
        return {"loss": total_loss}

    def validation_step(self, x, batch_idx):
        x = self.get_mod_cropped(x[0])
        recon, z = self.forward(x)
        
        regularization = 0.0
        if self.regularization == "l1":
            regularization = torch.abs(z).sum()
        elif self.regularization == "l2":
            regularization = torch.square(z).sum()

        recon_loss = self.reconstruction_loss(x, recon)
        total_loss = recon_loss + self.alpha * regularization

        self.log("val_loss", total_loss, prog_bar=True)
        return {"loss": total_loss}

    def test_step(self, x, batch_idx):
        x = self.get_mod_cropped(x[0])
        recon, z = self.forward(x)
        
        regularization = 0.0
        if self.regularization == "l1":
            regularization = torch.abs(z).sum()
        elif self.regularization == "l2":
            regularization = torch.square(z).sum()

        recon_loss = self.reconstruction_loss(x, recon)
        total_loss = recon_loss + self.alpha * regularization

        self.log("test_loss", total_loss, prog_bar=True)
        return {"loss": total_loss}

    def on_train_epoch_end(self, *args, **kwargs):
        if self.current_epoch % 5 == 0:
            self.encoder.eval()
            self.decoder.eval()
            test_batch = self.get_mod_cropped(self.test_loader.to(self.device))
            train_batch = self.get_mod_cropped(self.train_loader.to(self.device))
            with torch.no_grad():
                recon_test, _ = self.forward(test_batch)
                recon_train, _ = self.forward(train_batch)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay, amsgrad=True)

    def reconstruction_loss(self, x, recon):
        return self.loss_func(x, recon)

if __name__ == "__main__":
    Batch_size = 64
    NUM_epoch = 100
    rows = 28

    model = AE(
        mod_name="mnist",
        enc=MnistEncoder(latent_dim=LATENT_DIM, deterministic=True),
        dec=MnistDecoder(latent_dim=LATENT_DIM),
        rows=rows,
        lr=1e-3
    )

    base_dir = "./ae_mnist_rows"
    repo_name_base = "mnist_auto_encoder_crop"
    LATENT_DIM = 16  # 确保 LATENT_DIM 与模型结构一致

    # 遍历每个 crop 文件夹
    for crop_folder in sorted(os.listdir(base_dir)):
        version_dir = os.path.join(base_dir, crop_folder, "version_0", "checkpoints")

        if os.path.exists(version_dir):
            # 查找 .ckpt 文件
            ckpt_files = [f for f in os.listdir(version_dir) if f.endswith(".ckpt")]
            if len(ckpt_files) == 1:
                ckpt_path = os.path.join(version_dir, ckpt_files[0])

                # 为每个模型创建一个唯一的仓库名称
                crop_number = crop_folder.split('_')[-1]
                repo_name = f"{repo_name_base}_{crop_number}"

                # 加载模型
                model = AE(
                    mod_name="mnist",
                    enc=MnistEncoder(latent_dim=LATENT_DIM, deterministic=True),
                    dec=MnistDecoder(latent_dim=LATENT_DIM),
                    rows=28,  # 你可以根据实际情况动态设置 rows
                    lr=1e-3
                )

                # 加载权重
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['state_dict'])  # 假设 .ckpt 文件中包含 state_dict

                # 保存模型权重并上传到 Hugging Face
                model.save_pretrained(f"./{repo_name}")  # 本地保存
                model.push_to_hub(repo_name)  # 上传到 Hugging Face

                print(f"Uploaded {repo_name} to Hugging Face Hub")

    print("All models have been uploaded.")
