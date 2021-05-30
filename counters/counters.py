import torch
from counters.ssim import ssim


def mse(img1, img2):
    return (img1 - img2) ** 2


def psnr(img1, img2):
    return 10.0 * torch.log10(1.0 / (1e-5 + mse(img1, img2)))


def SSIMCounter(preds, grounds):
    scores = torch.zeros(preds.shape[0])
    for i in range(len(scores)):
        scores[i] = ssim(preds[i].unsqueeze(0), grounds[i].unsqueeze(0))
    return scores


def PSNRCounter(preds, grounds):
    return psnr(preds, grounds).reshape(preds.shape[0], -1).mean(1)


def MSECounter(preds, grounds):
    return mse(preds, grounds).reshape(preds.shape[0], -1).mean(1)


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
                # print("SHAPES")
                # print(len(names))
                # print(scores.shape)
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
