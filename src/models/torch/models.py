from collections import OrderedDict
from .utils import RunningLoss
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
import numpy as np

class LSTM(torch.nn.Module):
    def __init__(self, stateful=False, **config):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(**config)
        self.num_directions = (2 if config.get('bidirectional', False) else 1)
        self.linear = torch.nn.Linear(
            config['hidden_size'] * self.num_directions,
            config['input_size'])
        self.outs = {}
        self.cell_states = None

    def forward(self, input, initial=None):
        if not initial:
            out, (hn, cn) = self.lstm(input)
        else:
            out, (hn, cn) = self.lstm(input, initial)
        lin_out = self.linear(out[:, -1])
        self.outs = dict([('lstm_' + str(n), h) for n, h in enumerate(hn)])
        self.outs.update(linear=lin_out)
        self.cell_states = cn
        return lin_out

    def reset_states(self):
        self.outs = None
        self.cell_states = None

    def get_prev_states(self, batch_size):
        outs = self.outs
        if self.cell_states is None and not outs:
            return None
        keys = sorted(outs.keys())
        hn = []
        for key in keys:
            if 'lstm' in key:
                hn.append(outs[key].detach().numpy())

        cn = torch.tensor(self.cell_states)[:, :batch_size]
        hn = torch.tensor(np.array(hn))[:, :batch_size]

        return (hn, cn)

    def forecast(self, X, batch_size):
        """
        If a model stateful, then i-th example of next batch wiil be
        use the hidden and cell states from i-th example from previous batch
        """
        self.eval()
        pred = torch.zeros((0, X[0].shape[1]))
        for i in tqdm(range(0, len(X), batch_size)):
            sz = min(len(X)-i, batch_size)
            inp = torch.tensor(X[i:i+sz]).float()
            states = self.get_prev_states(sz)
            pred = torch.cat((pred, self.forward(inp, states)), dim=0)
        return pred


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device,
                 log_dir=None, stateful=False, hparams=None):
        self.sw = SummaryWriter(log_dir)
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.initial_epoch = 0
        self.device = device
        self.stateful = stateful
        self.epochs = 0
        self.hparams = hparams

    def forward(self, inputs):
        if not self.stateful or self.model.cell_states is None:
            return self.model(inputs)

        prev = self.model.get_prev_states(inputs.size(0))
        return self.model(inputs, prev)

    def pass_one_epoch(self, data, epoch, is_train):
        model = self.model
        sw = self.sw
        optimizer = self.optimizer

        self.running_loss = RunningLoss()
        if is_train:
            model.train()
        else:
            model.eval()
            outputs, grads = {}, {}

        with tqdm(data) as pbar:
            pbar.set_description('Epoch ' + str(epoch) + ' of ' +
                                 ('train: ' if is_train else 'val: '))
            for inputs, true in pbar:
                inputs = inputs.to(self.device)
                true = true.to(self.device)

                if is_train:
                    optimizer.zero_grad()

                output = self.forward(inputs)
                loss = self.criterion(output, true)

                pbar.set_postfix(loss=loss.item())
                self.running_loss.update(loss.item(), inputs.size(0))

                loss.backward()
                if is_train:
                    optimizer.step()
                else:
                    for name, out in model.outs.items():
                        outputs[name] = outputs.get(name, [])
                        outputs[name].extend(out.view(-1).detach().numpy())

                    for name, param in model.named_parameters():
                        grads[name] = grads.get(name, [])
                        grads[name].extend(param.grad.view(-1).numpy())

        if is_train:
            sw.add_scalar('Loss/train', self.running_loss.avg(), epoch)
        else:
            sw.add_scalar('Loss/test', self.running_loss.avg(), epoch)
            for name in outputs:
                sw.add_histogram('Outputs ' + name, np.array(outputs[name]),
                                 epoch)

            for name in grads:
                sw.add_histogram('Grads ' + name, np.array(grads[name]), epoch)

    def train(self, data, val_data, num_epochs):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e9

        for epoch in range(num_epochs):
            epoch += self.epochs

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            self.pass_one_epoch(data, epoch, is_train=True)
            self.model.reset_states()
            self.pass_one_epoch(val_data, epoch, is_train=False)
            self.model.reset_states()

            avg_loss = self.running_loss.avg()
            self.scheduler.step(avg_loss)
            
            print('Loss: {:.4f}'.format(avg_loss))

            # deep copy the model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

            print()
        
        self.best_loss = best_loss
        self.sw.add_hparams(self.hparams, {'best_loss': best_loss})

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        self.epochs += num_epochs
