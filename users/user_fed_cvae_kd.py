import copy

from torch.optim import SGD, Adam

from models.VAE import CVAE
from users.user import User
from utils import kl_divergence, one_hot_encode, reconstruction_loss


class UserFedCVAE(User):
    def __init__(self, base_params, z_dim, image_size, beta, data_amt, pmf, version):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.model = CVAE(
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            z_dim=z_dim,
            image_size=image_size,
            version=version,
        ).to(self.device)

        if base_params["use_adam"]:
            self.optimizer = Adam(self.model.parameters(), lr=base_params["local_LR"])
        else:
            self.optimizer = SGD(self.model.parameters(), lr=base_params["local_LR"])

        self.beta = beta
        self.data_amt = data_amt
        self.pmf = pmf

    def train(self, local_epochs):
        self.model.train()

        for epoch in range(local_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):

                y_hot = one_hot_encode(y_batch, self.num_classes)
                X_batch, y_hot = X_batch.to(self.device), y_hot.to(self.device)
                X_recon, mu, logvar = self.model(X_batch, y_hot, self.device)

                # Calculate losses
                recon_loss = reconstruction_loss(self.num_channels, X_batch, X_recon)  # 两张图像计算损失
                total_kld = kl_divergence(mu, logvar)  # 利用mu, logvar会生成Z,计算他们KL损失
                total_loss = recon_loss + self.beta * total_kld

                # Update net params
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def update_decoder(self, decoder_state_dict):
        """Helper method to swap out the current decoder for a new decoder ensuring it is a new object with a deep copy."""

        self.model.decoder.load_state_dict(copy.deepcopy(decoder_state_dict))
