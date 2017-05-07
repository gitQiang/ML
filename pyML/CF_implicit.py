# -*- coding: utf-8 -*-
## functions
## ref: https://github.com/benfred/implicit
## Fast Python Collaborative Filtering algorithm

def CF_implicit(ra_train):
    ## input format: UserID, ItemID, rates
    
    ## 第一步：source packages
    import implicit
    import pandas as pd
    from scipy.sparse import csr_matrix
    
    ## 第二步： 读取数据 
    ##ra_train = pd.read_table(filename, sep=sep, names=['UserID','ItemID','Rating'])
            
    ## 第三步： 创建稀疏矩阵
    # map each artist and user to a unique numeric value
    ra_train['UserID'] = ra_train['UserID'].astype("category")
    ra_train['ItemID'] = ra_train['ItemID'].astype("category")
    # create a sparse matrix of all the users/plays
    trainD = csr_matrix((ra_train['Rating'].astype(float), (ra_train['ItemID'].cat.codes.copy(), ra_train['UserID'].cat.codes.copy())))
            
    ### 第四步： 训练模型
    model = implicit.als.AlternatingLeastSquares()
    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(trainD)
            
    ## 第五步： 推荐商品或者用户
    # recommend items for a user
    # recommendations = model.recommend(userid=1, user_items=trainD.T, N=10)
    # print recommendations
    # find related items
    # related = model.similar_items(itemid=757)
    # print related
    return(model, trainD)
    
def oneusedexample():
    import pandas as pd
    ev0 = pd.read_csv("Data/****.csv")    
    ## split data into train and test datesets
    tsub0 = ev0['TX_DT'].apply(lambda x: x < '2016-12-18')
    tab0 = ev0.ix[tsub0,:]
    tsub1 = ev0['TX_DT'].apply(lambda x: x >= '2016-12-18')
    tab1 = ev0.ix[tsub1,:]
    
    ## training model
    ra_train = tab0[['ECIF_NUM','PROD_ID','TX_STAT_CD']]
    ra_train.columns = ['ItemID','UserID','Rating']
    ra_train.Rating = 1
    model, trainD = CF_3golden(ra_train)
    ra_test = tab1[['ECIF_NUM','PROD_ID','TX_STAT_CD']]
    ra_test.columns = ['ItemID','UserID','Rating']
    ra_test.Rating = 1

    ## predicting model for one product
    oneProd = '000D'
    one_test = ra_test.ix[ra_test.UserID==oneProd,:]

    import implicit
    Nsam = 5000
    prod_id = ra_train['UserID'].astype("category")
    int_id = prod_id.cat.codes.copy()
    userid = int_id[ra_train.UserID==oneProd]
    userid = userid.tolist()[0]
    ypred = model.recommend(userid=userid, user_items=trainD.T, N=Nsam)
    
    ### get performance for classifier 
    import numpy as np
    y = np.array([0]*Nsam)
    y = [any(one_test.ItemID==d) for d,p in ypred]
    y = [int(x) for x in y]
    #y = np.random.randint(0,2,Nsam).tolist()
    yprob = [p for d,p in ypred]
    yp = [int(d>0.5) for d in yprob]
    getMetrics(y, yp, yprob) ## under pyMetrics =====
    
    
if __name__ == '__main__':
    print 'add test example for CF_implicit'
    