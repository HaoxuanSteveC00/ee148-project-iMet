from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchsummary import summary
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
from iMetDataset import *
import matplotlib.pyplot as plt
import torchvision.models as M


'''
This code is adapted from last homework.
'''


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(6, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(42632, 1000) # 1 layer: 1352; 2 layer: 200; 3 layer: 8
        self.fc2 = nn.Linear(1000, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

    def forward_before_last_layer(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data, target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.cross_entropy(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        train_loss += loss.item()
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx * len(data) / len(train_loader.sampler), loss.item()))
    return train_loss / len(train_loader.sampler)


def validation(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))
    return test_loss


''' TODO
# Generate predictions
def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data in test_loader:
            data = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
'''



def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    parser.add_argument('--test-datasize', action='store_true', default=False,
                        help='train on different sizes of dataset')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    torch.manual_seed(args.seed)



    # Evaluate on the official test set
    # if args.evaluate:
    #     assert os.path.exists(args.load_model)
    #
    #     # Set the test model
    #     model = Net().to(device)
    #     model = M.resnet18(num_classes=99).to(device)
    #     model.load_state_dict(torch.load(args.load_model))
    #
    #     test_dataset = datasets.MNIST('./data', train=False,
    #                 transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ]))
    #
    #     test_loader = torch.utils.data.DataLoader(
    #         test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    #
    #     test(model, device, test_loader, analysis=True)
    #
    #     return




    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset_no_aug = TrainDataset(True, 'data/imet-2020-fgvc7/labels.csv',
                'data/imet-2020-fgvc7/train_20country.csv', 'data/imet-2020-fgvc7/train/',
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToPILImage(),           # Add data augmentation here
                    transforms.RandomResizedCrop(128),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
                ]))
    train_dataset_with_aug = train_dataset_no_aug
    assert(len(train_dataset_no_aug) == len(train_dataset_with_aug))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    np.random.seed(args.seed)
    subset_indices_valid = np.random.choice( len(train_dataset_no_aug), int(0.15*len(train_dataset_no_aug)), replace=False )
    subset_indices_train = [i for i in range(len(train_dataset_no_aug)) if i not in subset_indices_valid]
    # subset_indices_train = []
    # subset_indices_valid = []
    # for target in range(10):
    #     idx = (train_dataset_no_aug.targets == target).nonzero() # indices for each class
    #     idx = idx.numpy().flatten()
    #     val_idx = np.random.choice( len(idx), int(0.15*len(idx)), replace=False )
    #     val_idx = np.ndarray.tolist(val_idx.flatten())
    #     train_idx = [i for i in range(len(idx)) if i not in val_idx]
    #     subset_indices_train += np.ndarray.tolist(idx[train_idx])
    #     subset_indices_valid += np.ndarray.tolist(idx[val_idx])

    assert (len(subset_indices_train) + len(subset_indices_valid)) == len(train_dataset_no_aug)
    assert len(np.intersect1d(subset_indices_train,subset_indices_valid)) == 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset_with_aug, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset_no_aug, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )


    # Load your model [fcNet, ConvNet, Net]
    #model = Net().to(device)
    model = M.resnet18(num_classes=20).to(device)
    # summary(model, (1,28,28))

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # if args.test_datasize:
    #     train_final_loss = []
    #     val_final_loss = []
    #     train_size = []
    #     for i in [1, 2, 4, 8, 16]:
    #         print("Dataset with size 1/{} of original: ".format(i))
    #         subset_indices_train_sub = np.random.choice(subset_indices_train, int(len(subset_indices_train)/i), replace=False)
    #         train_loader_sub = torch.utils.data.DataLoader(
    #             train_dataset_with_aug, batch_size=args.batch_size,
    #             sampler=SubsetRandomSampler(subset_indices_train_sub)
    #         )
    #         train_losses = []
    #         val_losses = []
    #         for epoch in range(1, args.epochs + 1):
    #             train_loss = train(args, model, device, train_loader_sub, optimizer, epoch)
    #             val_loss = validation(model, device, val_loader)
    #             train_losses.append(train_loss)
    #             val_losses.append(val_loss)
    #             scheduler.step()    # learning rate scheduler
    #             # You may optionally save your model at each epoch here
    #         print("Train Loss: ", train_losses)
    #         print("Test Loss: ", val_losses)
    #         print("\n")
    #         train_final_loss.append(train_losses[-1])
    #         val_final_loss.append(val_losses[-1])
    #         train_size.append(int(len(subset_indices_train)/i))
    #
    #     plt.loglog(range(1, args.epochs + 1), train_losses)
    #     plt.loglog(range(1, args.epochs + 1), val_losses)
    #     plt.xlabel("Number of training examples")
    #     plt.ylabel("Loss")
    #     plt.legend(["Training loss", "Val loss"])
    #     plt.title("Training loss and val loss as a function of the number of training examples on log-log scale")
    #     plt.show()
    #     return

    # Training loop
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        val_loss = validation(model, device, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()    # learning rate scheduler
        # You may optionally save your model at each epoch here
        if args.save_model:
            torch.save(model.state_dict(), "mnist_model.pt")

    # plt.plot(range(1, args.epochs + 1), train_losses)
    # plt.plot(range(1, args.epochs + 1), val_losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend(["Training loss", "Val loss"])
    # plt.title("Training loss and val loss as a function of the epoch")
    # plt.show()



if __name__ == '__main__':
    main()
