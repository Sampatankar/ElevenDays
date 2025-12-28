# bytes vs str:
a = b'h\x65ll'
print(list(a))
print(a)

a = 'a\u0300 propos'
print(list(a))
print(a)


# Checking your default encoding system on your local environment:
import locale

print(locale.getpreferredencoding())


# An example failure when writing binary data to a file:
with open('data.bin', 'w') as f:
    f.write(b'\xf1\xf2\xf3\xf4\xf5')
# important to write using mode wb rather than mode w, in this case

