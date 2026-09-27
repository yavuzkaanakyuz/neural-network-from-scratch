import csv
import random

import torch
import torch.nn.functional as F


class Linear:
    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1, correct_3d=True):
        self.eps = eps
        self.momentum = momentum
        self.correct_3d = correct_3d
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            reduce_dims = (0, 1) if x.ndim == 3 and self.correct_3d else 0
            xmean = x.mean(reduce_dims, keepdim=True)
            xvar = x.var(reduce_dims, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, indices):
        self.out = self.weight[indices]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        batch, time, channels = x.shape
        x = x.view(batch, time // self.n, channels * self.n)
        self.out = x.squeeze(1) if x.shape[1] == 1 else x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.out = None

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def train(self):
        for layer in self.layers:
            layer.training = True

    def eval(self):
        for layer in self.layers:
            layer.training = False


def load_words(path):
    if str(path).endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as handle:
            words = [row["name"].strip().lower() for row in csv.DictReader(handle)]
    else:
        with open(path, encoding="utf-8") as handle:
            words = [line.strip().lower() for line in handle]
    return sorted({word for word in words if word and all(ch.isalpha() for ch in word)})


def prepare_data(words, block_size, seed=42):
    chars = sorted(set("".join(words)))
    stoi = {char: index + 1 for index, char in enumerate(chars)}
    stoi["."] = 0
    itos = {index: char for char, index in stoi.items()}
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)

    def build(items):
        xs, ys = [], []
        for word in items:
            context = [0] * block_size
            for char in word + ".":
                index = stoi[char]
                xs.append(context)
                ys.append(index)
                context = context[1:] + [index]
        return torch.tensor(xs), torch.tensor(ys)

    n1, n2 = int(0.8 * len(shuffled)), int(0.9 * len(shuffled))
    return {
        "train": build(shuffled[:n1]),
        "dev": build(shuffled[n1:n2]),
        "test": build(shuffled[n2:]),
        "stoi": stoi,
        "itos": itos,
    }


def build_mlp(vocab_size, block_size, n_embd=10, n_hidden=200, seed=2147483647):
    generator = torch.Generator().manual_seed(seed)
    model = Sequential([
        Embedding(vocab_size, n_embd, generator),
        Flatten(),
        Linear(block_size * n_embd, n_hidden, bias=False, generator=generator),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size, generator=generator),
    ])
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    return model


def build_wavenet(vocab_size, n_embd=10, n_hidden=68, correct_bn=True, seed=2147483647):
    generator = torch.Generator().manual_seed(seed)
    model = Sequential([
        Embedding(vocab_size, n_embd, generator),
        FlattenConsecutive(2),
        Linear(2 * n_embd, n_hidden, bias=False, generator=generator),
        BatchNorm1d(n_hidden, correct_3d=correct_bn),
        Tanh(),
        FlattenConsecutive(2),
        Linear(2 * n_hidden, n_hidden, bias=False, generator=generator),
        BatchNorm1d(n_hidden, correct_3d=correct_bn),
        Tanh(),
        FlattenConsecutive(2),
        Linear(2 * n_hidden, n_hidden, bias=False, generator=generator),
        BatchNorm1d(n_hidden, correct_3d=correct_bn),
        Tanh(),
        Linear(n_hidden, vocab_size, generator=generator),
    ])
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    return model


def count_parameters(model):
    return sum(parameter.nelement() for parameter in model.parameters())


def train_model(model, train_data, steps=30000, batch_size=64, seed=42):
    x_train, y_train = train_data
    parameters = model.parameters()
    for parameter in parameters:
        parameter.requires_grad = True
    generator = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for step in range(steps):
        indices = torch.randint(0, x_train.shape[0], (batch_size,), generator=generator)
        logits = model(x_train[indices])
        loss = F.cross_entropy(logits, y_train[indices])
        for parameter in parameters:
            parameter.grad = None
        loss.backward()
        learning_rate = 0.1 if step < int(steps * 0.75) else 0.01
        with torch.no_grad():
            for parameter in parameters:
                parameter -= learning_rate * parameter.grad
        history.append(loss.log10().item())
    return history


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    x, y = dataset
    return F.cross_entropy(model(x), y).item()


@torch.no_grad()
def sample(model, stoi, itos, block_size=8, count=20, seed=2147483657):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(count):
        context, output = [0] * block_size, []
        while len(output) < 30:
            logits = model(torch.tensor([context]))
            probabilities = F.softmax(logits, dim=1)
            index = torch.multinomial(probabilities, 1, generator=generator).item()
            if index == 0:
                break
            output.append(index)
            context = context[1:] + [index]
        names.append("".join(itos[index] for index in output))
    return names


def trace_shapes(model, x):
    rows = []
    for layer in model.layers:
        x = layer(x)
        rows.append((layer.__class__.__name__, tuple(x.shape)))
    return rows


def smooth_loss(history, group_size=1000):
    values = torch.tensor(history)
    usable = values[: values.numel() // group_size * group_size]
    return usable.view(-1, group_size).mean(1)
