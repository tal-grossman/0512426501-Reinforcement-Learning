import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def load_mnist(batch_size):
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader

def train(train_loader, num_epochs, optimizer, criterion, net):
    train_losses = []
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print loss for each batch
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        train_losses.append(running_loss / len(train_loader))
    return train_losses

def test(test_loader, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Neural Network Model
class Net_1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net_1, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out
    
class Net_2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net_2, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out
    
# Neural Network Model
class Net_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net_3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

NUM_EPOCHS = 100


def run_section_1():
    # Hyper Parameters
    input_size = 784
    num_classes = 10
    batch_size = 100
    learning_rate = 1e-3

    # Load MNIST dataset
    train_loader, test_loader = load_mnist(batch_size)

    net = Net_1(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # Train the Model
    train_losses = train(train_loader = train_loader, num_epochs = NUM_EPOCHS, optimizer = optimizer, criterion = criterion, net = net)

    # Test the Model
    accuracy = test(test_loader = test_loader, net = net)
    print('Accuracy of the network (section 1) on the 10000 test images: %d %%' % accuracy)

    # Save the Model and losses
    torch.save(net.state_dict(), 'model_part1.pth')
    np.save(f'./section1.npy', np.array(train_losses))

def run_section_2():
    # Hyper Parameters
    input_size = 784
    num_classes = 10
    batch_size = 100
    learning_rate = 0.01


    # Load MNIST dataset
    train_loader, test_loader = load_mnist(batch_size)


    net = Net_2(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # Train the Model
    train_losses = train(train_loader = train_loader, num_epochs = NUM_EPOCHS, optimizer = optimizer, criterion = criterion, net = net)

    # Test the Model
    accuracy = test(test_loader = test_loader, net = net)
    print('Accuracy of the network (section 2) on the 10000 test images: %d %%' % accuracy)

    # Save the Model and results
    torch.save(net.state_dict(), 'model_part2.pth')
    np.save(f'./section2.npy', np.array(train_losses))

def run_section_3():
    # Hyper Parameters
    input_size = 784
    hidden_size = 500
    num_classes = 10
    batch_size = 100
    learning_rate = 1e-3

    # Load MNIST dataset
    train_loader, test_loader = load_mnist(batch_size)

    net = Net_3(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Train the Model
    train_losses = train(train_loader, NUM_EPOCHS, optimizer, criterion, net)

    # Test the Model
    accuracy = test(test_loader, net)
    print('Accuracy of the network (section 3) on the 10000 test images: %d %%' % accuracy)

    # Save the Model
    torch.save(net.state_dict(), 'model_part3.pth')
    np.save(f'./section3.npy', np.array(train_losses))


def plot_saved_results():
    # Load results
    train_losses_1 = np.load('section1.npy')
    train_losses_2 = np.load('section2.npy')
    train_losses_3 = np.load('section3.npy')

    # Plots
    plt.plot(train_losses_1, color='blue')
    plt.plot(train_losses_2, color='green')
    plt.plot(train_losses_3, color='red')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Cross Entropy Loss', fontweight='bold')
    plt.title('Train Loss Vs Epochs', fontweight='bold')
    plt.legend(['Section 1', 'Section 2', 'Section 3'], loc='upper right')
    plt.annotate(f'Accuracy on validation:\nsection 1 = 89%\nsection 2 = 92%\nsection 3 = 98%', xy = (0.3, 0.8), xycoords='axes fraction')
    plt.savefig("train_loss_plots.png")
    plt.show()

def _get_parameters():
    parser = ArgumentParser()

    parser.add_argument('-p', '--plot_only', action='store_true', help='Only plot the saved results')
    return parser.parse_args()


if __name__ == "__main__":

    args = _get_parameters()
    if args.plot_only:
        plot_saved_results()
        exit()

    run_section_1()

    run_section_2()

    run_section_3()

    plot_saved_results()

