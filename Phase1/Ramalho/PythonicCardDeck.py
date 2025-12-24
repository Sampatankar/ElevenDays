import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self) -> None:
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]
    

# Representation of cards in the deck:
beer_card = Card('7', 'diamonds')
print(beer_card)

# How many cards in the deck:
deck = FrenchDeck()
print(len(deck))

# Get the first and last card from the deck:
print(f"The first card is:",deck[0],"and the last card is:",deck[-1])

# Three random card selections:
from random import choice
print(choice(deck), choice(deck), choice(deck))



