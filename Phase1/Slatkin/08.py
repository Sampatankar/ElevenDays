# enumerate vs zip:
names = ['Cecilia', 'Lise', 'Marie']
counts = [len(n) for n in names]
max_count = 0

for i, name in enumerate(names):
    count = counts[i]
    if count > max_count:
        longest_name = name
        max_count = count
    print(count)


for name, count in zip(names, counts):
    if count > max_count:
        longest_name = name
        max_count = count
    print(count)