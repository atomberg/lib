import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


# Hyper Parameters
input_size = 784
hidden_size = 500
n_classes = 10
n_epochs = 5
batch_size = 100
learning_rate = 3e-4


class DNN(nn.Module):
    def __init__(self, layer_widths, input_dim, output_dim):
        super().__init__()

        fc = []
        self.n_layers = len(layer_widths)
        for in_w, out_w in zip([input_dim] + layer_widths, layer_widths + [output_dim]):
            fc.append(nn.Linear(in_w, out_w))
        self.fc = fc

    def forward(self, x):
        activations = [F.relu for _ in self.fc[:-1]] + [F.sigmoid]
        for act, fc in zip(activations, self.fc):
            x = act(fc(x))
        return x


dnn = DNN(layer_widths=[30, 45, 67, 100], input_dim=input_size, output_dim=n_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28 * 28)).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = dnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}]'
                  f'Step [{i+1}/{len(train_dataset)//batch_size}]'
                  f'Loss: {loss.data[0]:.4f}')

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
