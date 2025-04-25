import numpy as np

class PolynomialPrediction:
    def __init__(self):
        self.polys = []
        self.t1 = 0.0  # start
        self.t2 = 0.0  # end

    def setPolynomial(self, pls):
        self.polys = pls

    def setTime(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def valid(self):
        return len(self.polys) == 3

    def evaluate(self, t):
        # t should be in [t1, t2]
        tv = np.array([1.0, t, t**2, t**3, t**4, t**5])
        
        pt = np.zeros(3)
        pt[0] = np.dot(tv, self.polys[0])
        pt[1] = np.dot(tv, self.polys[1]) 
        pt[2] = np.dot(tv, self.polys[2])

        return pt

    def evaluateConstVel(self, t):
        tv = np.array([1.0, t])
        
        pt = np.zeros(3)
        pt[0] = np.dot(tv, self.polys[0][:2])
        pt[1] = np.dot(tv, self.polys[1][:2])
        pt[2] = np.dot(tv, self.polys[2][:2])

        return pt
