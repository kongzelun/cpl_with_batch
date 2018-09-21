import os
import logging
# import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import models


def setup_logger(level=logging.DEBUG, filename=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


def main():
    logger = setup_logger(filename='log.txt')

    train_epoch_number = 10
    batch_size = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset = np.loadtxt(models.Config.dataset_path, delimiter=',')
    # np.random.shuffle(dataset[:5000])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = models.DataSet(dataset[:5000])
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=24)

    # testset = models.DataSet(dataset[5000:])
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=24)

    prototypes = {}

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=device, number_layers=8, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)

    gcpl = models.GCPLLoss(threshold=models.Config.threshold, gamma=models.Config.gamma, b=models.Config.threshold, tao=1.0, beta=0.5, lambda_=0.001)
    sgd = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if not os.path.exists("pkl"):
        os.mkdir("pkl")

    if os.path.exists(models.Config.pkl_path):
        state_dict = torch.load(models.Config.pkl_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file %s.", models.Config.pkl_path)
        except RuntimeError:
            logger.error("Loading state from file %s failed.", models.Config.pkl_path)

    for epoch in range(train_epoch_number):
        logger.info("Trainset size: %d, Epoch number: %d", len(trainset), epoch + 1)

        running_loss = 0.0

        for i, (features, labels) in enumerate(trainloader):
            features = features.to(net.device)
            sgd.zero_grad()
            features = net(features).view(batch_size, 1, -1)
            loss = gcpl(features, labels, prototypes)
            loss.backward()
            sgd.step()

            running_loss += loss.item() / batch_size

            logger.debug("[%3d, %5d] loss: %7.4f", epoch + 1, i + 1, loss.item() / batch_size)

        torch.save(net.state_dict(), models.Config.pkl_path)

        prototype_count = 0

        for c in prototypes:
            prototype_count += len(prototypes[c])

        logger.info("Prototypes Count: %d", prototype_count)

        # if (epoch + 1) % 5 == 0:
        distance_sum = 0.0
        correct = 0

        for i, (feature, label) in enumerate(testloader):
            feature = net(feature.to(net.device)).view(1, -1)
            predicted_label, probability, min_distance = models.predict(feature, prototypes)

            if label == predicted_label:
                correct += 1

            distance_sum += min_distance

            logger.debug("%5d: Label: %d, Prediction: %d, Probability: %7.4f, Distance: %7.4f, Accuracy: %7.4f",
                         i + 1, label, predicted_label, probability, min_distance, correct / (i + 1))

        logger.info("Distance Average: %7.4f", distance_sum / len(testloader))
        logger.info("Accuracy: %7.4f\n", correct / len(testloader))


if __name__ == '__main__':
    main()
