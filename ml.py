import torch

def ConvertLabeledTemplatesToTensors(labeled_templates):
    template_shape = labeled_templates[0]["template"].shape
    num_templates = len(labeled_templates)
    x = torch.empty(num_templates, *template_shape)
    y = torch.zeros(num_templates, loader.kNumClasses)
    for i, labeled_template in enumerate(labeled_templates):
        x[i] = labeled_template["template"]
        y[i] = labeled_template["class"]

    return x.to_device(), y.to_device()

def TrainClassifier(classifier, train_x, train_y, val_x, val_y):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters())
    for epoch in range(20):
        train_loss = 0
        val_loss = 0
        batch_size = 32

        train_batches = train_x.size() // batch_size
        for i in range(train_batches):
            batch_x = train_x[i * batch_size : (i + 1) * batch_size]
            batch_y = train_y[i * batch_size : (i + 1) * batch_size]
            batch_z = classifier(batch_x)

            optimizer.zero_grad()
            loss = criterion(batch_z, batch_y)
            optimizer.step()
            train_loss += loss.item()
            print('[epoch: %d, batch: %d] train_loss: %.3f' % (epoch + 1, i + 1, loss / (i + 1)))

        val_batches = val_x.size() // batch_size
        for i in range(val_batches):
            batch_x = val_x[i * batch_size : (i + 1) * batch_size]
            batch_y = val_y[i * batch_size : (i + 1) * batch_size]
            batch_z = classifier(batch_x)

            loss = critierion(batch_z, batch_y)
            optimizer.step()
            val_loss += loss.item()
            print('[epoch: %d, validation batch: %d] val_loss: %.3f' % (epoch + 1, i + 1, loss / (i + 1)))


# TODO change the classifier to take in grayscale images
class Dense2DClassifier(torch.nn.Module):
    def __init__(self, num_classes, input_size):
        super(Dense2DClassifier, self).__init__()
        self.hidden_layer_size = int((input_size[0] * input_size[1]) ** .5)

        self.linear_1 = torch.nn.Linear(numpy.prod(input_size), self.hidden_layer_size)
        self.linear_2 = torch.nn.Linear(self.hidden_layer_size, num_classes)


    def forward(self, x):
        x = torch.flatten(x)
        x = self.linear_1(x).clamp(min = 0, max = 6)
        x = self.linear_2(x)
        x = torch.nn.Softmax(x)
        return x
