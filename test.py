class A:
    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, score):
        self.__score = score

class B(A):
    def test(self, s):
        self.score = s
        print(self.score)

b = B()
b.test(100)