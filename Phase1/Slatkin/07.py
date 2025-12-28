# Range vs enumerate:
flavour_list = ['vanilla', 'chocolate', 'pecan', 'strawberry']

for i in range(len(flavour_list)):
    flavour = flavour_list[i]
    print(f'{i + 1}: {flavour}')


for i, flavour in enumerate(flavour_list, 1):
    print(f'{i}: {flavour}')