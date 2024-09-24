class Outer :
    def __init__(self, number):
        self.number = number
        self.inner = self.Inner(self)

    class Inner :
        def __init__(self, outer):
            self.outer = outer
            self.number = outer.number + 10

outer = Outer(5)

print(outer.inner.number)  # Output: 15