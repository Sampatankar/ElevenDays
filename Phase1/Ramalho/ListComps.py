# without using list comprehensions:
symbols = '$@£|€±'
codes = []
for symbol in symbols:
    codes.append(ord(symbol))

print(codes)


# with list comprehension:
symbols = '$@£|€±'
codes = [ord(symbol) for symbol in symbols]
print(codes)


# Listcomps vs Map & Filter:
# Listcomp:
beyond_ascii = [ord(s) for s in symbols if ord(s) > 127]
print(beyond_ascii)

# Map & Filter:
beyond_ascii = list(filter(lambda c: c > 127, map(ord, symbols)))
print(beyond_ascii)


# Cartesian Products:
colours = ['black', 'white']
sizes = ['S', 'M', 'L', 'XL']
tshirts = [(colour, size) for colour in colours for size in sizes]
print(tshirts)