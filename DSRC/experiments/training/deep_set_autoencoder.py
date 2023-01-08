from DSRC.experiments.models.bennu_particle_ejection_return import EncoderNetwork

import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


class DecoderNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input = torch.nn.Linear(EncoderNetwork.OUTPUT_SIZE, 25)
        self.activation0 = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(25, 10)
        self.activation1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(10, EncoderNetwork.INPUT_SIZE)
        self.activation2 = torch.nn.ReLU()

    def forward(self, inp):
        out = self.input(inp)
        out = self.activation0(out)
        out = self.hidden1(out)
        out = self.activation1(out)
        out = self.hidden2(out)
        out = self.activation2(out)

        return out

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = EncoderNetwork()
        self.decoder = DecoderNetwork()

    def forward(self, inp):
        out = self.encoder(inp)
        out = self.decoder(out)

        return out


def generate_single_input(_):
    fuel_level = np.random.uniform(low=0.1, high=1.0)
    sample_value = np.random.uniform(0.0, 10.0)
    time_delta = np.random.normal()
    position = np.random.uniform(low=-500, high=500, size=(3,))

    ob = (fuel_level, sample_value, time_delta, *position)

    #return torch.Tensor(ob, device=torch.device('cuda'))
    return ob


def generate_batch(batch_size, pool):
    return pool.map(generate_single_input, range(batch_size))
    


if __name__ == '__main__':
    model = Autoencoder().cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5)

    num_epochs = int(7e6)
    batch_size = 526

    model.train()

    with mp.Pool() as pool:
        for epoch in (pbar := tqdm(range(num_epochs))):
            obs = generate_batch(batch_size, pool)
            data = torch.cuda.FloatTensor(obs)
            output = model(data)
            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch: {epoch+1}/{num_epochs}, loss: {loss:10.3}")

    torch.save(model.state_dict(), "./model.pth")
