import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from functions import one_hot
import pandas as pd


class VAEs(nn.Module):
    def __init__(self, dropout=0.1, kernal=5, stride=1, n_dis=10): # kernal~k-mer
        super().__init__()
        self.latent_distributions = n_dis
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            nn.Conv1d(20, 40, kernal, stride),
            nn.ReLU(),
            nn.Conv1d(40, 80, kernal, stride),
            nn.ReLU(),
            nn.Conv1d(80, 160, kernal, stride),
            nn.ReLU(),
            nn.Conv1d(160, 160, kernal, stride, bias=True),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(160, 160, kernal, stride, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(160, 80, kernal, stride),
            nn.ReLU(),
            nn.ConvTranspose1d(80, 40, kernal, stride),
            nn.ReLU(),
            nn.ConvTranspose1d(40, 20, kernal, stride),
            nn.Sigmoid(),
        )

    # log normal distribution
    def reparameterization(self, z_mean, z_log_var):
        # sampling Z from μ + σ(eps), eps~N(0,1)
        eps = torch.randn(z_mean.size())
        # st_dev = var^(1/2) = exp(log(var^(1/2))) = exp(1/2 * log(var))
        z = z_mean + eps * torch.exp(z_log_var/2)
        return z

    # encoder
    def recognition(self, input):
        x = self.dropout(self.encoder(input))
        n, linear_in = x.shape[0], x.shape[1]
        mean_linear = nn.Linear(linear_in, self.latent_distributions)
        log_var_linear = nn.Linear(linear_in, self.latent_distributions)
        means, log_vars = mean_linear(x), log_var_linear(x)
        z = self.reparameterization(means, log_vars)
        return n, linear_in, z, means, log_vars

    #decoder
    def forward(self, input):
        n, linear_out, z, means, log_vars = self.recognition(input)
        reconstruct_linear = nn.Linear(self.latent_distributions, linear_out)
        decoder_in = reconstruct_linear(z).reshape(n, 160, int(linear_out/160))
        x_hat = self.decoder(decoder_in)
        return means, log_vars, x_hat


if __name__ == "__main__":
    # import sequences
    good_seqs = pd.read_csv('All_affibody_greater_10.csv')
    input_seqs = list(good_seqs['Sequence'])

    model = VAEs()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    # checking number of parameters in model
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    trainloader = torch.utils.data.DataLoader(input_seqs, batch_size=30, shuffle=True)
    # plot_x = []
    # plot_y = []
    for epoch in range(30):  # loop over the dataset
        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            x = one_hot(data).transpose(1, 2)  # shape (n, channel, seq_len)
            # print(data.shape)

            # zero the parameter gradients
            optimizer.zero_grad()
            # running model
            means, log_vars, x_hat = model(x.float())

            # total loss = reconstruction loss + KL divergence
            reconstruc_loss = torch.sum(torch.square(x.float() - x_hat))
            kl_div = 0.5 * torch.sum(torch.exp(log_vars) + means**2 - 1 - log_vars, axis=1) # sum over latent dimension
            loss = (reconstruc_loss + kl_div).sum() # maximizing the lower bound

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update parameters

            # print loss statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                # lot_x.append('[%d, %5d]' % (epoch + 1, i + 1))
                # plot_y.append(running_loss/100)
                running_loss = 0.0
    print('Finished Training')

