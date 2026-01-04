"""
The WizCoin class, represents a fictional currency and the number of coins in that class.  

The denominations are knuts, sickles (worth 29 knuts) and galleons (worth 17 sicles or 493 knuts).

"""

class WizCoin:
    def __init__(self, galleons, sickles, knuts) -> None:
        """
        Create a new WizCoin object with galleons, sickles and knuts
        """
        self.galleons = galleons
        self.sickles = sickles
        self.knuts = knuts
        # Note: __init__() methods never have a return statement

    def value(self):
        """
        The value in knuts of all coins in this WizCoin Object

        """
        return (self.galleons * 17 * 29) + (self.sickles * 29) + (self.knuts)
    
    def weightInGrams(self):
        """Returns the weight of the coins in grams"""
        return (self.galleons * 31.03) + (self.sickles * 11.34) + (self.knuts * 5.0)


