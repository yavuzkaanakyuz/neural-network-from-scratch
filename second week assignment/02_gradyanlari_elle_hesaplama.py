import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def tanh(self):
        return Value(math.tanh(self.data), (self,), "tanh")


# 1. örnek: L = (a*b+c)*f
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
d = e + c
f = Value(-2.0, label="f")
L = d * f

# türevleri burada elle yazdım
L.grad = 1.0
d.grad = f.data
f.grad = d.data
c.grad = d.grad
e.grad = d.grad
a.grad = b.data * e.grad
b.grad = a.data * e.grad

print("Basit ifade:", {"a": a.grad, "b": b.grad, "c": c.grad, "f": f.grad})


# 2. örnek: iki girişli tek nöron
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
bias = Value(6.8813735870195432, label="b")
x1w1 = x1 * w1
x2w2 = x2 * w2
n = x1w1 + x2w2 + bias
o = n.tanh()

# chain rule: dış türev ile yerel türevi çarpıyorum
o.grad = 1.0
n.grad = (1 - o.data**2) * o.grad
bias.grad = n.grad
x1w1.grad = n.grad
x2w2.grad = n.grad
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

print("Nöron çıktısı:", o.data)
print("Nöron gradyanları:", {"x1": x1.grad, "w1": w1.grad, "x2": x2.grad, "w2": w2.grad, "b": bias.grad})

