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

# Look at the top three cards from a new deck and pick just the aces by starting on index 12 and skipping 13 cards at a time:
print(f"The top three cards of a brand new deck:",deck[:3])
print(f"Picking aces by starting on index 12 and skipping 13 cards at a time:",deck[12::13])

# Forward iteration:
for card in deck:
    print(f"The forward iteration:", card)

# Reversed iteration:
for card in reversed(deck):
    print(f"The reversed deck:", card)

# Ranking and sorting the cards:
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card) -> int:
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
    print(f"The sorted deck:",card)

