import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += power * self.data ** (power - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def backward(self):
        topo, visited = [], set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


degerler = [2.0, 0.0, -3.0, 1.0, 6.8813735870195432]
x1, x2, w1, w2, b = [Value(v) for v in degerler]
n = x1 * w1 + x2 * w2 + b
e = (2 * n).exp()
o = (e - 1) / (e + 1)  # tanh'ı parçaladım
o.backward()
micrograd_sonuc = [o.data, x1.grad, x2.grad, w1.grad, w2.grad, b.grad]


def normal_hesap(v):
    x1, x2, w1, w2, b = v
    n = x1 * w1 + x2 * w2 + b
    e = math.exp(2 * n)
    return (e - 1) / (e + 1)


# geçen haftadaki sayısal türev
h = 1e-6
sayisal = []
for i in range(len(degerler)):
    arti, eksi = degerler.copy(), degerler.copy()
    arti[i] += h
    eksi[i] -= h
    sayisal.append((normal_hesap(arti) - normal_hesap(eksi)) / (2 * h))


try:
    import torch
except ImportError as hata:
    raise SystemExit("PyTorch gerekli: pip install torch") from hata

tx1, tx2, tw1, tw2, tb = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in degerler]
to = torch.tanh(tx1 * tw1 + tx2 * tw2 + tb)
to.backward()
pytorch_sonuc = [to.item(), tx1.grad.item(), tx2.grad.item(), tw1.grad.item(), tw2.grad.item(), tb.grad.item()]

print("               çıktı, x1, x2, w1, w2, b")
print("backward():   ", [round(v, 8) for v in micrograd_sonuc])
print("sayısal türev:", [round(normal_hesap(degerler), 8)] + [round(v, 8) for v in sayisal])
print("PyTorch:      ", [round(v, 8) for v in pytorch_sonuc])

for a, b in zip(micrograd_sonuc[1:], sayisal):
    assert abs(a - b) < 1e-6
for a, b in zip(micrograd_sonuc, pytorch_sonuc):
    assert abs(a - b) < 1e-9

print("Üç yöntem de eşleşti.")

