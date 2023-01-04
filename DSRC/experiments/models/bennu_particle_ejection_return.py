"""PyTorch model for Bennu Particle Ejection Sample Return experiment.

The model includes a 'deep set' taken from the states of other craft
in the simulation on once.
"""

from DSRC.experiments.bennu_particle_ejection_return import ActionSpace

import torch

# Since we intend to use genetic optimization there is no need
# to perform gradient calculations
torch.set_grad_enabled(False)


class EncoderNetwork(torch.nn.Module):
    """Encoder network.

    The encoder network is used to process each input
    element of the set. The result of the encoder network
    for each element is aggregated using a permutation-invarient
    operation, e.g., addition, for later processing.
    """

    INPUT_SIZE = 6
    OUTPUT_SIZE = 40

    def __init__(self):  # noqa D
        super().__init__()

        self.input = torch.nn.Linear(EncoderNetwork.INPUT_SIZE, 10)
        self.activation0 = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(10, 25)
        self.activation1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(25, EncoderNetwork.OUTPUT_SIZE)
        self.activation2 = torch.nn.ReLU()

    def forward(self, inp):
        """Perform forward-pass through model."""
        out = self.input(inp)
        out = self.activation0(out)
        out = self.hidden1(out)
        out = self.activation1(out)
        out = self.hidden2(out)
        out = self.activation2(out)

        return out


class DecoderNetwork(torch.nn.Module):
    """Decoder Network.

    The decoder network takes the aggregated output of
    each input's encoded value (given by passing that input
    through the encoder network) and is used to calculate
    the Q value.
    """

    INPUT_SIZE = EncoderNetwork.OUTPUT_SIZE + 3  # includes the mothership position
    OUTPUT_SIZE = ActionSpace.num_states()  # is 25

    def __init__(self):  # noqa D
        super().__init__()

        self.input = torch.nn.Linear(DecoderNetwork.INPUT_SIZE, 30)
        self.activation0 = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(30, DecoderNetwork.OUTPUT_SIZE)
        self.activation1 = torch.nn.Softmax(dim=0)

    def forward(self, inp):
        """Perform forward-pass through the model."""
        out = self.input(inp)
        out = self.activation0(out)
        out = self.hidden1(out)
        out = self.activation1(out)

        return out


class Model(torch.nn.Module):
    """Q function approximator model."""

    def __init__(self):  # noqa D
        super().__init__()

        self.encoder_net = EncoderNetwork()
        self.decoder_net = DecoderNetwork()

    def forward(self, obs, mothership_pos):
        """Take a forward pass given the observations."""
        states_inp = [torch.Tensor(o) for o in obs]
        states_enc = [self.encoder_net(s) for s in states_inp]

        states_repr = torch.nn.functional.normalize(sum(states_enc), dim=0)
        mship_pos = torch.Tensor(mothership_pos)

        qnet_inp = torch.cat((states_repr, mship_pos))

        return self.decoder_net(qnet_inp)

    def get_action(self, obs, mothership_pos):
        """Get the action enum value."""
        out = self.forward(obs, mothership_pos)
        return ActionSpace.vals(int(torch.argmax(out)))
