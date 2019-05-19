import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(32)))


def compute_accuracy(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv5_bn = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv6_bn = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(32, 48, 3, padding=(0, 0))
        self.conv7_bn = nn.BatchNorm2d(48)

        self.conv8 = nn.Conv2d(48, 32, 1, padding=(0, 0))
        self.conv8_bn = nn.BatchNorm2d(32)

        self.conv9 = nn.Conv2d(32, 16, 1, padding=(0, 0))
        self.conv9_bn = nn.BatchNorm2d(16)

        self.global_avg_pool = nn.AvgPool2d(5)

        self.fc10 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2), stride=2)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2), stride=2)

        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = F.relu(x)

        x = self.conv9(x)
        x = self.conv9_bn(x)
        x = F.relu(x)

        x  = self.global_avg_pool(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc10(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
skip=False
for epoch in range(10):  # loop over the dataset multiple times
    print_every = 200
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i % print_every) == (print_every-1):
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
            running_loss = 0.0
        if skip:
            break

    # Print accuracy after every epoch
    accuracy = compute_accuracy(net, testloader)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))

    if skip:
        break

print('Finished Training')


dataiter = iter(testloader)
images, labels = dataiter.next()
images,  labels = images.to(device), labels.to(device)

# print images
imshow(torchvision.utils.make_grid(images).cpu())
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images,  labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images,  labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


