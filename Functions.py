import math


class Functions:

    def __init__(self, price_series, volume_series, time=None):
        self.price = price_series
        self.volume = volume_series
        self.current_time = time

    def set_time(self,time):
        self.current_time = time

    @staticmethod
    def add(x, y): return x + y

    @staticmethod
    def sub(x, y): return x - y

    @staticmethod
    def norm(x, y): return abs(x-y)

    @staticmethod
    def mul(x, y): return x * y

    @staticmethod
    def div(x, y):
        if y!= 0:
            return x / y
        else:
            return math.inf

    @staticmethod
    def and_f(x, y): return x and y

    @staticmethod
    def or_f(x, y): return x or y

    @staticmethod
    def not_f(x): return not x

    @staticmethod
    def larger(x, y): return x > y

    @staticmethod
    def smaller(x, y): return x < y

    @staticmethod
    def if_then_else(x, y, z):
        if x:
            return y
        else:
            return z

    def average(self, p_v, n):
        if p_v:
            return (self.price[self.current_time - n]+self.price[ self.current_time]) / n
        else:
            return (self.volume[self.current_time - n]+self.volume[ self.current_time]) / n

    def max(self, p_v, n):
        if p_v:
            return max(self.price[self.current_time - n],self.price[ self.current_time])
        else:
            return max(self.volume[self.current_time - n],self.volume[ self.current_time])

    def min(self, p_v, n):
        if p_v:
            return min(self.price[self.current_time - n], self.price[self.current_time])
        else:
            return min(self.volume[self.current_time - n],self.volume[ self.current_time])

    def lag(self, p_v, n):
        if p_v:
            return self.price[self.current_time - n]
        else:
            return self.volume[self.current_time - n]

    def volatility(self, n):
        avg = sum(self.price) / len(self.price)
        return sum((x - avg) ** 2 for x in self.price) / len(self.price)
