import numpy as np
from sklearn.ensemble import RandomForestRegressor
train_feat=np.genfromtxt("train_feat.txt",dtype=np.float32)
train_id=np.genfromtxt("train_id.txt",dtype=np.float32)
test_feat=np.genfromtxt("test_feat.txt",dtype=np.float32)
rf = RandomForestRegressor()
rf.fit(train_feat, train_id)
print(rf.predict(test_feat))
for i in rf.predict(test_feat):
    print(i)

#[ 1.]
#[ 1.  1.9]
"""
data2=[[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]]
target2=[0,1,2,3,4,5]
rf2 = RandomForestRegressor()
rf2.fit(data2, target2)
print(rf2.predict([[1,1,1,1,1]]))
"""
'''
data=[[0,0,0],[1,1,1],[2,2,2],[1,1,1],[2,2,2],[0,0,0]]
target=[2,2,2,1,2,0]
'''
