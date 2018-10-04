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
    pred = np.squeeze(pred_raw.cpu().numpy(), axis=0)
    feature = pred_raw.type(torch.FloatTensor) / N_CLASSES

    turn_logit = classifier(feature)
    l = [label]
    l = torch.tensor(l).type(torch.LongTensor)
    loss = F.nll_loss(turn_logit, l)
    loss.backward()
    optimizer.step()
    print(loss.detach().numpy(), label, turn_logit.detach().cpu().numpy())

def train(args):
    device = "cpu"

    # Setup model
    model = get_model({"arch":"fcn8s"}, N_CLASSES, version="mit_sceneparsing_benchmark")
    state = convert_state_dict(torch.load(args.model_path, map_location='cpu')["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Setup classifier
    classifier = Classifier()
    classifier.eval()
    classifier.to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=0.0001, momentum=True)

    print("Read Input csv file from : {}".format(args.csv_path))
    with open(args.csv_path) as csv_file:
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
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()
            for i in range(20):
                train_step(model, classifier, optimizer, img, int(row['label']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )

    parser.add_argument(
        "--csv_path", nargs="?", type=str, default=None, help="Path of the input csv file"
    )
    args = parser.parse_args()
    train(args)
