def matrixNormalize(X, method='colsum'):
    ## X : an np array
    ## method: rowsum, colsum, rowmax, colmax, rowmean, colmean
    
    import numpy as np
    if method == 'colmax':
        Xnew = X / X.max(axis=0)
    
    if method == 'rowmax':
        Xnew = X / X.max(axis=1)
    
    if method == 'colmean':
        Xnew = X / X.mean(axis=0)
    
    if method == 'rowmean':
        Xnew = X / X.mean(axis=1)
        
    if method == 'rowsum':
        tmp = X.sum(axis=1)
        Xnew = X / tmp[:, np.newaxis]
    
    if method == 'colsum':
        tmp = X.sum(axis=0)
        Xnew = X / tmp[np.newaxis,:]
        
    return(Xnew)

if __name__ == '__main__':
    ## test example here
    import numpy as np
    X = np.array([[1000,  10,   0.5],
              [ 765,   5,  0.35],
              [ 800,   7,  0.09]])
    print "Raw numpy array:"
    print X
    print "column sum normalize:"
    print matrixNormalize(X, method='colsum')
    print "row sum normalize:"
    print matrixNormalize(X, method='rowsum')
    print "column max normalize:"
    print matrixNormalize(X, method='colmax')
    print "row max normalize:"
    print matrixNormalize(X, method='rowmax')
    print "column mean normalize:"
    print matrixNormalize(X, method='colmean')
    print "row mean normalize:"
    print matrixNormalize(X, method='rowmean')
    
    # print "please test example for this function"
    
    