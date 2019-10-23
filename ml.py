import torch
import numpy as np
import util

def ConvertLabeledTemplatesToTensors(labeled_templates, num_classes):
    template_shape = labeled_templates[0]["template"].shape
    num_templates = len(labeled_templates)
    x = torch.empty(num_templates, *template_shape)
    y = torch.zeros(num_templates, dtype = torch.long)
    for i, labeled_template in enumerate(labeled_templates):
        x[i] = torch.from_numpy(labeled_template["template"])
        y[i] = labeled_template["class"]

    return x.cuda(), y.cuda()

def InverseClassDistribution(y, num_classes):
    distribution = torch.zeros(num_classes)
    for v in y:
        distribution[v] += 1
    distribution = distribution / torch.sum(distribution)
    return torch.ones(num_classes) / distribution

def ApproximatelyEqual(a, b):
    return abs(a - b) < abs(a + b) / 1000

dist = InverseClassDistribution(torch.tensor([0, 0, 1]), 2)
assert(ApproximatelyEqual(dist[1], dist[0] * 2))

def TrainClassifier(classifier, train_x, train_y, val_x, val_y):
    # Pass class weights here
    criterion = torch.nn.CrossEntropyLoss(weight = InverseClassDistribution(train_y, classifier.num_classes).cuda())
    optimizer = torch.optim.SGD(classifier.parameters(), lr = .01, momentum = 0.9)
    for epoch in range(75):
        train_loss = 0
        val_loss = 0
        batch_size = 32

        total = 0
        correct = 0
        train_batches = train_x.size()[0] // batch_size
        for i in range(train_batches):
            batch_x = train_x[i * batch_size : (i + 1) * batch_size]
            batch_y = train_y[i * batch_size : (i + 1) * batch_size]

            optimizer.zero_grad()
            batch_z = classifier(batch_x)
            _, predicted = torch.max(batch_z, 1)
            total += batch_x.size()[0]
            correct += (predicted == batch_y).sum().item()
            loss = criterion(batch_z, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[epoch: %d, batch: %d] train_loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, train_loss / (i + 1), correct / total))

        total = 0
        correct = 0
        val_batches = val_x.size()[0] // batch_size
        for i in range(val_batches):
            batch_x = val_x[i * batch_size : (i + 1) * batch_size]
            batch_y = val_y[i * batch_size : (i + 1) * batch_size]
            batch_z = classifier(batch_x)

            _, predicted = torch.max(batch_z, 1)
            total += batch_x.size()[0]
            correct += (predicted == batch_y).sum().item()
            loss = criterion(batch_z, batch_y)
            val_loss += loss.item()
            print('[epoch: %d, validation batch: %d] val_loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, val_loss / (i + 1), correct / total))

# TODO change the classifier to take in grayscale images
class Dense2DClassifier(torch.nn.Module):
    def __init__(self, num_classes, input_size):
        super(Dense2DClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_layer_size = int((input_size[0] * input_size[1]) ** .5)

        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(np.prod(input_size), self.hidden_layer_size)
        self.linear_2 = torch.nn.Linear(self.hidden_layer_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_1(x).clamp(min = 0, max = 6)
        x = self.linear_2(x)
        return x
