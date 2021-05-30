import torch
from torch.distributions import Categorical

softmax = torch.nn.Softmax(dim=3)
cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="none")


def Entropy(outs, grounds=None):
    outs = outs.transpose(1, 3)
    maxed = softmax(outs)
    entropy = Categorical(probs=maxed).entropy()
    return entropy.sum(1).sum(1)


def EntropyMax(preds, grounds, th=0.6):
    mask = 1 - (grounds == 255).int()
    outs = preds.transpose(1, 3)
    entropy = Categorical(logits=outs).entropy()
    entropy = entropy * mask
    return entropy.max(1)[0].max(1)[0]


def EntropyTH(preds, grounds, th=0.6):
    mask = 1 - (grounds == 255).int()
    outs = preds.transpose(1, 3)
    entropy = Categorical(logits=outs).entropy()
    entropy = entropy * mask
    return (entropy > th).sum(1).sum(1)


def EntropyNoBackground(outs, grounds=None):
    mask = torch.where(grounds > 0, 1, 0)
    outs = outs.transpose(1, 3)
    maxed = softmax(outs)
    entropy = Categorical(probs=maxed).entropy()
    entropy = entropy * mask
    return entropy.sum(1).sum(1)


def CrossEntropy(
    outs,
    grounds=None,
):
    cross_ent = cross_entropy(outs, grounds)
    return cross_ent.sum(1).sum(1)


def CrossEntropyNoBackground(outs, grounds=None):

    mask = torch.where(grounds > 0, 1, 0)
    cross_ent = cross_entropy(outs, grounds)
    cross_ent = cross_ent * mask
    return cross_ent.sum(1).sum(1)


class Counter:
    def __init__(self, metric, name, save_name):
        self.metric = metric
        self.name = name
        self.counter = dict()
        self.save_name = save_name

    def update(self, outs, names, grounds):
        scores = self.metric(outs, grounds)
        scores = scores.detach().cpu().numpy()
        for i in range(len(names)):
            if names[i] in self.counter:
                self.counter[names[i]] = self.counter[names[i]] + scores[i]
            else:
                self.counter[names[i]] = scores[i]

    def save(self, epoch):
        save_path = (
            f'{self.save_name.replace(".txt","")}_{self.name}_ep_{epoch}.txt'
        )
        with open(save_path, "w") as f:
            for name in self.counter:
                f.write(f"{name} {self.counter[name]}\n")


class CounterIterator:
    def __init__(self, save_path):
        self.counters = dict()
        self.save_path = save_path

    def add(self, func, name):
        self.counters[name] = Counter(func, name, self.save_path)

    def update(self, outs, names, grounds):
        for name in self.counters:
            self.counters[name].update(outs, names, grounds)

    def save(self, epoch):
        for name in self.counters:
            self.counters[name].save(epoch)
