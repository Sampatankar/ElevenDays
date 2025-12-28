# % formatting operator:
a = 0b10111011
b = 0xc5f
print("Binary is %d, hex is %d" % (a, b))


# Advanced formatting through the built-in str.format function:
a = 1234.5678
formatted = format(a, ',.2f')
print(formatted)

b = 'my string'
formatted = format(b, '^20s')
print('*', formatted, '*')

# with placeholders:
key = 'my_var'
value = 1.234

formatted = '{:<10} = {:.2f}'.format(key, value)
print(formatted)


# F-strings:
places = 3
number = 1.23456

print(f"My number is {number:.{places}f}")
"""
F-strings are a succinct method to allow for arbitrary Python expressions to be directly embedded within format specifiers
"""

