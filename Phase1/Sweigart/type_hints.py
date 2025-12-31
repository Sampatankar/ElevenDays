# Example of static typing in Python:
def describeNumber(number: int) -> str:
    if number % 2 == 1:
        return 'An odd number. '
    elif number == 42:
        return 'The answer. '
    else:
        return 'Yes, that is the number. '
    

myLuckyNumber: int = 42
print(describeNumber(myLuckyNumber))