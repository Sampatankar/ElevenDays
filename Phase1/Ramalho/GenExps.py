# Initialising a tuple and an array from a genexp:
import array

symbols = '$@£|€±'
this_tuple = tuple(ord(symbol) for symbol in symbols)
print(this_tuple)

this_array = array.array('I', (ord(symbol) for symbol in symbols))
print(this_array)


# Cartesian products in a generator expression: 
# (%s calls the str(value) method in Python, and here we code to substitute %s with a variable value such as colour)
colours = ['black', 'white']
sizes = ['S', 'M', 'L', 'XL']
for tshirt in ('%s %s' % (colour, size) for colour in colours for size in sizes):
    print(tshirt)