import torch
import argparse
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict

N_CLASSES = 151

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(900, 3)

    def forward(self, x):
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, 900)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def decode_segmap(temp, plot=False):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, N_CLASSES):
        r[temp == l] = 10 * (l % 10)
        g[temp == l] = l
        b[temp == l] = 0

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def train_step(feature_net, classifier, optimizer, img, label):
    optimizer.zero_grad()
    outputs = feature_net(img)
    pred_raw = outputs.data.max(1)[1]
    feature = pred_raw.type(torch.FloatTensor) / N_CLASSES

    turn_logit = classifier(feature)
    l = torch.tensor(label).type(torch.LongTensor)
    loss = F.nll_loss(turn_logit, l)
    loss.backward()
    optimizer.step()
    print(loss.detach().numpy())
    # print(label)
    # print(turn_logit.detach().cpu().numpy())

def eval(feature_net, classifier, img, label):
    outputs = feature_net(img)
    pred_raw = outputs.data.max(1)[1]
    feature = pred_raw.type(torch.FloatTensor) / N_CLASSES
    turn_logit = classifier(feature)
    print("accuracy", (turn_logit.max(1)[1] == torch.tensor(label)).sum().double() / len(label))

def read_samples(csv_path, batch_size):
    images = []
    labels = []
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            img = misc.imread(row['image'])
            img = misc.imresize(img, (240, 240))
            img = img[:, :, ::-1]
            img = img.astype(np.float64)
            img -= np.array([104.00699, 116.66877, 122.67892])
            img = img.astype(float) / 255.0

            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            # img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()
            images.append(img)
            labels.append(int(row['label']))

    permutation = torch.randperm(len(images))
    batches = []
    for i in range(0, len(images), batch_size):
        batches.append((torch.stack(images[i:i+batch_size]), labels[i:i+batch_size]))
    return batches


def train(args):
    device = "cpu"

    # Setup model
    model = get_model({"arch":"fcn8s"}, N_CLASSES, version="mit_sceneparsing_benchmark")
    state = convert_state_dict(torch.load(args.feature_model_path, map_location='cpu')["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Setup classifier
    classifier = Classifier()
    if args.classifier_model_path is not None:
        classifier.load_state_dict(torch.load(args.classifier_model_path, map_location='cpu'))

    classifier.to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=True)

    if args.train_csv_path is not None:
        print("Read training csv file from : {}".format(args.train_csv_path))
        train_data = read_samples(args.train_csv_path, args.batch_size)
        for i in range(args.num_epoch):
            for img, label in train_data:
                train_step(model, classifier, optimizer, img, label)
        torch.save(classifier.state_dict(), args.output_model_path)

    if args.test_csv_path is not None:
        classifier.eval()
        print("Read testing csv file from : {}".format(args.test_csv_path))
        test_data = read_samples(args.test_csv_path, 999)
        eval(model, classifier, test_data[0][0], test_data[0][1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument("--feature_model_path", nargs="?", type=str, help="Path to the saved feature model"
    )

    parser.add_argument("--classifier_model_path", nargs="?", type=str, help="Path to the saved classifier model"
    )

    parser.add_argument(
        "--output_model_path", nargs="?", type=str, default=None, help="Path to save the trained model"
    )

    parser.add_argument(
        "--train_csv_path", nargs="?", type=str, default=None, help="Path of the training csv file"
    )

    parser.add_argument(
        "--test_csv_path", nargs="?", type=str, default=None, help="Path of the testing csv file"
    )

    parser.add_argument(
        "--batch_size", nargs="?", type=int, default=1, help="training batch size"
    )

    parser.add_argument(
        "--num_epoch", nargs="?", type=int, default=1, help="number of epochs to train"
    )

    args = parser.parse_args()
    train(args)
