class A():
    def __init__(self, a, b, c=1, d=2):
        print(a, b, c, d)

class B(A):
    def __init__(self, e, *args):
        super(B, self).__init__(*args)
        print(e)

b = B(3, 2, 3, c = 3, d = 4)
