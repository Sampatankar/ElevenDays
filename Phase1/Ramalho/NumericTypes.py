"""
Implementing a class that represents 2D vectors
(Doing this rather than using the built-in complex type so that we can extend to represent n-dimensional vectors)

The Vector class models a 2D mathematical vector and teaches:
- How operators (+, *, abs, bool) are syntax sugar
- How Python translates that syntax into special ("dunder") methods
- How you define semantics, not Python

The hypot import computes the square root of (x squared plus y squared), enabling computation of vector magnitude

The __init__ creates the vector x, y and sets defaults

The __repr__ rather just pointing to a memory block, allows visual representation of the output

The __abs__ calculates the magnitude of the vector as a float value scalar

The __bool__ returns true or false depending on whether the vector is zero or non-zero

The __add__ does not mutate either vector, returns a new vector and mirrors mathematical vector addition

The __mul__ allows for vectors and numbers to be multiplied, returning a new vector in the same direction with multiplied magnitude


"""


# From the book:
from math import hypot

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)
    
    def __abs__(self):
        return hypot(self.x, self.y)
    
    def __bool__(self):
        return bool(abs(self))
    
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)


# Design of the API:
v1 = Vector(2, 4)
v2 = Vector(2, 1)
print(v1 + v2)

# Using abs in our API to calculate the magnitude of a vector:
v = Vector(3, 4)
print(abs(v))

# Using the * operator to multiply a vector with a number(scalar):
print(v * 3)
print(abs(v * 3))

