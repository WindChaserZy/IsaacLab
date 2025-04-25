import math

class Vector3i:
    def __init__(self, x :int = 1, y: int = 0, z :int = 0):
        self.x :int = x
        self.y :int = y
        self.z :int = z
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Invalid index")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Invalid index")

    def __repr__(self):
        return f"Vector3i({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        return isinstance(other, Vector3i) and \
               self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        return Vector3i(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3i(self.x - other.x, self.y - other.y, self.z - other.z)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __call__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        return self

    def __mul__(self, other):
        if isinstance(other, Vector3i):
            # Inner product
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif isinstance(other, int):
            # Scalar multiplication
            return Vector3i(self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Unsupported operand for *: 'Vector3i' and '{}'".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, k):
        self.x *= k
        self.y *= k
        self.z *= k
        return self

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def norm2(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2
    
    def flatten(self):
        return [self.x, self.y, self.z]
    
    def __hash__(self):
        seed = 0
        data = self.flatten()
        for elem in data:
            h = hash(elem)
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2)
        return seed
