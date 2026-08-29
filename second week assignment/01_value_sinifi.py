class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self._prev = set(_children)  # beni oluşturan değerler
        self._op = _op  # hangi işlemden çıktım
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")


# videodaki ilk ifade
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e + c
d.label = "d"

print("d:", d)
print("d işlemi:", d._op)
print("d'yi oluşturanlar:", d._prev)

