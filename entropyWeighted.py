def entropyWeighted(X):
    import numpy as np
    row_sums = X.sum(axis=0)
    P = X / row_sums[np.newaxis,:]
    
    k = 1/np.log(X.shape[0])
    E = np.multiply(P, np.log(P))
    ev = -1 * k * E.sum(axis=0)
    delv = 1.0 - ev
    wv = delv/np.sum(delv)
    Iv = np.dot(P, wv)
    
    return(Iv)

if __name__ == '__main__':
    ## test example here
    import numpy as np
    X = np.array([[1000,  10,   0.5, 10],
                  [ 765,   5,  0.35, 20],
                  [ 800,   7,  0.09, 40],
                  [ 870,   20,  0.332, 45],
                  [ 405,   12,  0.123, 23]])
    
    print entropyWeighted(X)
    print "complete: please test example for this function"
    