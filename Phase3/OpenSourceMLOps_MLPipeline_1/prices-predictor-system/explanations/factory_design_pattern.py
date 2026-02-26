from abc import ABC, abstractmethod


# Step 1: Define the Product interface
class Coffee(ABC):
    @abstractmethod
    def prepare(self):
        pass


# Step 2: Implement Concrete Products
class Espresso(Coffee):
    def prepare(self):
        return "Preparing a rich and strong Espresso."


class Latte(Coffee):
    def prepare(self):
        return "Preparing a smooth and creamy Latte."


class Cappuccino(Coffee):
    def prepare(self):
        return "Preparing a frothy Cappuccino."


# Step 3: Implement the Factory (CoffeeMachine)
class CoffeeMachine:
    def make_coffee(self, coffee_type):
        if coffee_type == "Espresso":
            return Espresso().prepare()
        elif coffee_type == "Latte":
            return Latte().prepare()
        elif coffee_type == "Cappuccino":
            return Cappuccino().prepare()
        else:
            return "Unknown coffee type!"


# Step 4: Use the Factory to Create Products
if __name__ == "__main__":
    machine = CoffeeMachine()

    coffee = machine.make_coffee("Espresso")
    print(coffee)  # Output: Preparing a rich and strong Espresso.

    coffee = machine.make_coffee("Latte")
    print(coffee)  # Output: Preparing a smooth and creamy Latte.

    coffee = machine.make_coffee("Cappuccino")
    print(coffee)  # Output: Preparing a frothy Cappuccino.


"""
In summary:
The code separates three concerns:
1. What a coffee is supposed to do -> Coffee (the interface/contract)
2. How specific coffees do it -> Espresso, Latte, Cappuncino
3. Deciding which coffee to make -> CoffeeMachine (the factory)

The calling code doesn't need to know how coffee is prepared - it just asks the machine

Using the Abtract Base Class (ABC) - Coffee is not meant to be instantiated, it just defines the contract: any class that claims to be a Coffee must implement prepare().  Any subclass of Coffee is forced to implement prepare(), which prevents bugs at runtime by enforcing the interface.

It also prevents instantiating the Coffee baseclass which is incomplete, which would produce a coffee with no recipe. 

It enables Polymorphism as every factory can treat them all the same, while each subclass provides its own behaviour.

The factory creates objects, with the creation logic centralised in one place.  The caller doesn't need to know about constructors, e.g. the caller never sees 'Latte()'.

When the code calls on the factory to make a coffee, it doesn't care what class is used, how it is prepared, only cares about the result.

We use the @abstractmethod so that if there are errors they occur immediately.  The method enforces that prepare contract. This is important for data ingestion pipelines, ML model interfaces, plugin systems and large teams/long-lived codebases.
"""