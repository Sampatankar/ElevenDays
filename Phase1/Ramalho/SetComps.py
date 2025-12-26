# Example of building a set of Latin-1 characters that have the word 'SIGN' in their Unicode names:

from unicodedata import name

set_output = {chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i), '')}
print(set_output)

