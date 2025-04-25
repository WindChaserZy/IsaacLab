import math

class Vector3d:
    def __init__(self, x=1.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
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
        return f"Vector3d({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        return isinstance(other, Vector3d) and \
               self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)

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
        if isinstance(other, Vector3d):
            # Inner product
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return Vector3d(self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Unsupported operand for *: 'Vector3d' and '{}'".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return Vector3d(self.x / other, self.y / other, self.z / other)
        else:
            raise TypeError("Unsupported operand for /: 'Vector3d' and '{}'".format(type(other)))

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero") 
            self.x /= other
            self.y /= other
            self.z /= other
            return self
        else:
            raise TypeError("Unsupported operand for /=: 'Vector3d' and '{}'".format(type(other)))

    def __imul__(self, k):
        self.x *= k
        self.y *= k
        self.z *= k
        return self

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def norm2(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def normalize(self):
        k = self.norm()
        if k != 0:
            self.x /= k
            self.y /= k
            self.z /= k

    def setZero(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
    def head2(self):
        return Vector3d(self.x, self.y, 0.0)
    
    def flatten(self):
        return [self.x, self.y, self.z]
    
    def normalized(self):
        k = self.norm()
        if k != 0:
            return Vector3d(self.x / k, self.y / k, self.z / k)
        else:
            return Vector3d(0.0, 0.0, 0.0)
    
