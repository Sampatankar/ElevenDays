# Using default arguments to reduce argument complexity:
def introduction(name, greeting="Hello"):
    print(greeting + ', ' + name)

introduction("Alice")


# Using * to pass in varying numbers of positional arguments:
def product(*args):
    result = 1
    for num in args:
        result *= num
    return result

print(product(3, 3))
print(product(2, 1, 2, 3))


# Using ** to pass in varying numbers of keyword arguments:
def formMolecules(**kwargs):
    if len(kwargs) == 2 and kwargs['hydrogen'] == 2 and kwargs['oxygen'] == 1:
        return 'water'
    # (rest of the function code would go here)

print(formMolecules(hydrogen=2, oxygen=1))


# Using * and ** to create wrapper functions:
def printLower(*args, **kwargs):
    args = list(args)
    for i, value in enumerate(args):
        args[i] = str(value.lower())
    return print(*args, **kwargs)

name = 'Albert'
printLower('Hello', name)
printLower('DOG', 'CAT', 'MOOSE', sep=', ')


# Standard function vs lambda function:
def rectanglePerimeter(rect):
    return print((rect[0] * 2) + (rect[1] * 2))

myRectangle = [4, 10]
rectanglePerimeter(myRectangle)

rectanglePerimeterL = lambda rect: (rect[0] * 2) + (rect[1] * 2)
print(rectanglePerimeterL([4, 10]))
