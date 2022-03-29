import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from functions import one_hot


class VAEs(nn.Module):
    def __init__(self, dropout=0.1, kernal=5, n_dis=10): # kernal~k-mer
        super().__init__()
        self.latent_distributions = n_dis
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            nn.Conv1d(20, 40, kernal, stride=1),
            nn.ReLU(),
            nn.Conv1d(40, 80, kernal, stride=2),
            nn.ReLU(),
            nn.Conv1d(80, 160, kernal, stride=2),
            nn.ReLU(),
            nn.Conv1d(160, 160, kernal, stride=1, bias=True),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(160, 160, kernal, stride=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(160, 80, kernal, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(80, 40, kernal, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(40, 20, kernal, stride=1),
            nn.Sigmoid(),
        )

    def reparameterization(self, z_mean, z_var):
        # sampling Z from μ + σ(eps), eps~N(0,1)
        eps = torch.randn(z_mean.size())
        z = z_mean + eps * torch.sqrt(z_var)
        return z

    # encoder
    def recognition(self, input):
        x = self.dropout(self.encoder(input))
        n, linear_in = x.shape[0], x.shape[1]
        mean_linear = nn.Linear(linear_in, self.latent_distributions)
        var_linear = nn.Linear(linear_in, self.latent_distributions)
        means, vars = mean_linear(x), var_linear(x)
        z = self.reparameterization(means, vars)
        return n, linear_in, z, means, vars

    #decoder
    def forward(self, input):
        n, linear_out, z, means, vars = self.recognition(input)
        reconstruct_linear = nn.Linear(self.latent_distributions, linear_out)
        decoder_in = reconstruct_linear(z).reshape(n, 160, int(linear_out/160))
        x_hat = self.dropout(self.decoder(decoder_in))
        return z, means, vars, x_hat


if __name__ == "__main__":
    seqs = ['AEAKYAEENCNACCSICPLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
            'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
    data = one_hot(seqs).transpose(1, 2)  # shape (n, channel, seq_len)
    print(data.shape)

    model = VAEs()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    z, means, vars, x_hat = model(data.float())
    print(x_hat.shape)

    # total loss = reconstruction loss + KL divergence
    reconstruc_loss = torch.sum(torch.square(data-x_hat))
    kl_div = 0.5 * torch.sum(vars + means ** 2 - 1 - torch.log(vars), axis=1)  # sum over latent dimension
    loss = reconstruc_loss + kl_div

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # Update parameters



