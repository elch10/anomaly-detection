import copy
from collections import OrderedDict
from itertools import zip_longest

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

class TimeDistributed(torch.nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class LSTM(torch.nn.Module):
    def __init__(self,
                 stateful,
                 linear_transform,
                 inp_stddev=0.2,
                 layers_stddev=0,
                 **config):
        """
        Linear transform specifies whether to add linear layer
        to transform from hidden_size of LSTM to desired output_size
        inp_stddev specifies standard deviatino of gaussian noise added to input
        layers_stddev is analogue of inp_stddev but for between layer connections
        """
        super(LSTM, self).__init__()

        self.linear_transform = linear_transform
        self.stateful = stateful
        self.inp_stddev = inp_stddev
        self.layers_stddev = layers_stddev

        config = copy.deepcopy(config)
        self.num_layers = config.pop('num_layers', 1)
        self.num_directions = (2 if config.get('bidirectional', False) else 1)
        output_size = config.pop('output_size', config['input_size'])

        inp_sizes = [config.pop('input_size')] + [config.get('hidden_size') * self.num_directions] * (self.num_layers - 1)

        self.lstm = torch.nn.ModuleList()
        for sz in inp_sizes:
            self.lstm.append(torch.nn.LSTM(input_size=sz, **config))

        self.linear_transform = linear_transform
        if linear_transform:
            self.td = TimeDistributed(torch.nn.Linear(
                    config['hidden_size'] * self.num_directions,
                    output_size),
                config['batch_first'])

        self.outs = {}
        self.cell_states = None

    def add_gaussian(self, inp, stddev):
        if stddev > 0 and self.training:
            return inp + torch.autograd.Variable(torch.randn(inp.size()) * stddev)
        return inp

    def lstm_forward(self, input, initial=None):
        out = input
        hns, cns = [], []
        
        if not initial:
            initial = []

        for i, (lstm, init) in enumerate(zip_longest(self.lstm, zip(*initial))):
            out, (hn, cn) = lstm(out, init)
            if i != len(self.lstm) - 1 or self.linear_transform:
                out = self.add_gaussian(out, self.layers_stddev)
            hns.append(hn)
            cns.append(cn)

        return out, (hns, cns)


    def forward(self, input, initial=None, last_only=False):
        """
        Last_only characterizes whether to return value only
        in time stamp t of LSTM or return sequence of timestamps
        """
        input = self.add_gaussian(input, self.inp_stddev)

        out, (hn, cn) = self.lstm_forward(input, initial)

        self.outs = OrderedDict([('lstm_' + str(n), h) for n, h in enumerate(hn)])
        self.cell_states = cn

        if last_only:
            out = out[:, -1]

        if self.linear_transform:
            out = self.td(out)
            self.outs.update(linear=out)
        
        return out

    def reset_states(self):
        self.outs = None
        self.cell_states = None

    def get_prev_states(self, batch_size):
        outs = self.outs
        if self.cell_states is None and not outs:
            return None

        keys = outs.keys()
        hn = []
        for key in keys:
            if 'lstm' in key:
                hn.append(outs[key][:, :batch_size].detach())

        cn = [0] * len(self.cell_states)
        for i in range(len(self.cell_states)):
            cn[i] = self.cell_states[i][:, :batch_size].detach()

        return (hn, cn)

    def forecast(self, X, batch_size):
        """
        Forecasts value at next time, having previous windows_len examples
        If a model stateful, then i-th example of next batch wiil be
        use the hidden and cell states from i-th example from previous batch
        """
        self.eval()
        pred = torch.zeros((0, self.td.module.out_features))
        for i in tqdm(range(0, len(X), batch_size)):
            sz = min(len(X)-i, batch_size)
            inp = torch.tensor(X[i:i+sz]).float()
            states = self.get_prev_states(sz)
            pred = torch.cat((pred, self.forward(inp, states, True)), dim=0)
        return pred



class RunningLoss:
    def __init__(self):
        self.losses = []

    def update(self, loss, size):
        self.losses += [loss] * size

    def avg(self):
        return np.mean(self.losses)



class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, 
                 device, log_dir, hparams):
        self.sw = SummaryWriter(log_dir)
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.stateful = model.stateful
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
                sw.add_histogram('Outputs/' + name, np.array(outputs[name]),
                                 epoch)

            for name in grads:
                sw.add_histogram('Grads/' + name, np.array(grads[name]), epoch)

    def train(self, data, val_data, num_epochs):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e9

        for epoch in range(num_epochs):
            epoch += self.epochs

            print(f'Epoch {epoch}/{self.epochs + num_epochs - 1}')
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
