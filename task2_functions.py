import torch
import matplotlib.pyplot as plt
from torch import nn
from numpy.random import default_rng

# class definitions
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

class CNeuralNetwork(nn.Module):
    def __init__(self):
        super(CNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(16, 256, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 512, 7, stride=1, padding=1, padding_mode='reflect'),
            nn.MaxPool2d(3),
        )
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

# Convenience functions
def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

def test(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
@torch.no_grad()
def explore_wrong_5x5(dataloader, model, device, class_labels=None, seed=None):
    model.eval()
    rng = default_rng(seed)
    all_wrong = torch.empty(0, dtype=torch.int64, device=device)
    preds = torch.empty(0, dtype=torch.int64, device=device)
    gtruths = torch.empty(0, dtype=torch.int64, device=device)
    for X, y in dataloader:
        X = X.to(device)
        pred = model(X).argmax(1)
        y = y.to(device)
        wrong = pred != y
        wrong_ixs = torch.argwhere(wrong).flatten()
        for ix in wrong_ixs:
            all_wrong = torch.cat((all_wrong, X[ix, ...]))
            preds = torch.cat((preds, torch.tensor([pred[ix]]).to(device)))
            gtruths = torch.cat((gtruths, torch.tensor([y[ix]]).to(device)))
    
    example_ixs = rng.choice(range(len(gtruths)), 25, replace=False)
    
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(14, 14))
    fig.tight_layout()
    for i, ix in enumerate(example_ixs):
        X = all_wrong[ix]
        y = gtruths[ix]
        y_guess = preds[ix]
        if class_labels:
            true = class_labels[y]
            guess = class_labels[y_guess]
        else:
            true = str(y)
            guess = str(y_guess)
        ax = axes.flatten()[i]
        ax.set_title(f'True:{true}, Guess:{guess}')
        im = X.squeeze().cpu()
        ax.imshow(im, cmap='gray')
    model.train()