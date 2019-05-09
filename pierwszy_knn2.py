import numpy as np
from sortedcontainers import SortedList
from util import get_data
from datetime import datetime

class KNN():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    @staticmethod
    def calc_dist_squared(x, xt):
        diff = x -xt
        d = diff.dot(diff)
        return d

    @staticmethod
    def create_point(distance, value):
        point = (distance, value)
        return point

    @staticmethod
    def is_list_shorter_than_k(sorted_list, k):
        return len(sorted_list) < self.k

    @staticmethod
    def prev_dist(sorted_list):
        return sorted_list[-1][0]

    @staticmethod
    def is_distance_shorter_than_last(distance, sorted_list):
        return d < prev_dist()

    @staticmethod
    def delete_last(sorted_list):
        del sorted_list[-1]
        return sorted_list

    def predict(self, X):
        y = np.zeros(len(X))
        print ("predict starting")
        for i, test_x_sample in enumerate(X):
            sorted_list = SortedList() #load=self.k

            for j, train_x_sample in enumerate(self.train_X):
                dist_squared = KNN.calc_dist_squared(test_x_sample, train_x_sample)
                value = self.train_y[j]
                point = KNN.create_point(dist_squared, value)

                if KNN.is_list_shorter_than_k:
                   sorted_list.add(point)
                else:
                   if KNN.is_distance_shorter_than_last(distance, sorted_list):
                      KNN.delete_last(sorted_list)
                      sorted_list.add(point)

            votes = {}
            for _, v in sorted_list:
                votes[v] = votes.get(v,0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                   max_votes = count
                   max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y):
         P = self.predict(X)
         return np.mean(P == Y)

if __name__ == '__main__':
     X, Y = get_data(2000)
     Ntrain = 1000
     Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
     Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
     for k in (1,2,3,4,5):
         print ("k=",k)
         knn = KNN(k)
         t0 = datetime.now()
         knn.fit(Xtrain,Ytrain)
         t1 = datetime.now()

         score = knn.score(Xtest, Ytest)
         t2 = datetime.now()
         print ("train", t1-t0, "test", t2-t1, "result", score)
