def disTransform(x, tran=1):
    import numpy as np
    x = np.array(x)

    from scipy.stats import zscore
    def sigmoid_array(x):
        return 1 / (1 + np.exp(-x))
    
    if tran == 1:
        xnew = zscore(x)
    
    if tran == 2:
        xnew = sigmoid_array(x)
    
    if tran == 3:
        xnew = (x - np.min(x)) / (np.max(x) - np.min(x))
        
    if tran == 4:
        xnew = (np.max(x) - x) / (np.max(x) - np.min(x))
        
    if tran == 5:
        xnew = np.log(x)
        
    if tran == 6:
        xnew = np.log10(x)
        
    if tran == 7:
        xnew = 1/np.abs(x - np.mean(x))
    
    return(xnew)
    