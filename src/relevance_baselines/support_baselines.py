'''
Created on Sep 25, 2014

@author: liorab
'''

def supp_baseline():
    """
    P(s is relevant to c)* [a*sentiment match +b*semantic match]...
    Stages:
    1. First calc the the p(rel) =  using 
                                1.1 exp(-KLdiv)
                                1.2 RM score :score(s) = score (R;s) - the score a sentence got with RM3
                                1.3 semantic similarity
    2. For each feature (sentiment similarity etc), for each claim s, convert the value to prob by dividing by the sum across all the sen.
     
    """

def main():
    supp_baseline()

if __name__ == '__main__':
    main() 