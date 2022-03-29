import torch
import torch.nn as nn
from functions import one_hot

# for dim_embedding expansion
class Overcomplete_Autoencoder(nn.Module):
    def __init__(self, in_dim, bias=True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim*2, bias),
            nn.LeakyReLU(),
            nn.Linear(in_dim*2, in_dim*4, bias),
            nn.LeakyReLU(),
            nn.Linear(in_dim*4, in_dim*8, bias),
            nn.LeakyReLU(),
            nn.Linear(in_dim*8, in_dim*16, bias),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_dim*16, in_dim*8, bias),
            nn.LeakyReLU(),
            nn.Linear(in_dim*8, in_dim*4, bias),
            nn.LeakyReLU(),
            nn.Linear(in_dim*4, in_dim*2, bias),
            nn.LeakyReLU(),
            nn.Linear(in_dim*2, in_dim, bias),
        )

    def forward(self, input):
        z = self.encoder(input)
        out = torch.sigmoid(self.decoder(z))
        return z, out

if __name__ == '__main__':
    seqs = ['AEAKYAEENCNACCSICPLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
            'AEAKYAEENCNACCSICSLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK']
    data = one_hot(seqs)
    print(data.shape)
    model = Overcomplete_Autoencoder(in_dim=20)
    out = model(data)
    print(out)
