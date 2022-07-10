from torch.nn.modules.activation import Tanh
import model_base
import torch.nn as nn
import torch

MSELoss = nn.MSELoss()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(1, 50, num_layers=3)
        self.tanh = nn.Tanh()
        self.lin = nn.Linear(50, 1, False)

    def forward(self, x):
        x = x[..., None]
        output = self.lstm(x)[0]
        output = self.tanh(output)
        output = self.lin(output)
        return output.squeeze(-1)

model = Model()
optim = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_f = MSELoss

model.load_state_dict(torch.load('50model.pt', map_location=torch.device('cpu')))
#model_base.run_model(model, optim, loss_f)
#torch.save(model.state_dict(), 'lrmodel.pt')
model_base.view_hist(model)