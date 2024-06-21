import random, bisect


class Zipfian:
    def __init__(self, alpha):
        self.alpha = alpha
        self.maxnoise = 1024

        
        self.accprob = []
        cumprob, prob = 0, 0

        for i in range(1, self.maxnoise):
            prob = 1.0 / pow(i, alpha)
            cumprob += prob
            self.accprob.append(cumprob)
        
        self.largeprob = self.accprob[-1]
        # print(self.accprob)

    
    def Generate(self):
        randness = random.uniform(0, self.largeprob)
        randnum = bisect.bisect_left(self.accprob, randness) + 1
        return randnum

n = 10000000
B = 1024
alpha = 1

zipf = Zipfian(alpha)
zipflist = [[zipf.Generate()] for i in range(n)]
# print (zipflist)
file = open("zipf,alpha=" + str(alpha) + ".txt", "w")
file.write(str(n) + '\n')
file.write(str(B) + '\n')
for line in zipflist:
    file.write(str(line[0]) + '\n')

# import matplotlib as mpl 
# import numpy as np
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt

# print("plot")
# plt.hist(zipflist, bins = B)
# plt.show()