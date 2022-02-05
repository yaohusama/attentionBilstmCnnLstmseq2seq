import pandas as pd
import scipy
from scipy import io
#第一，要加载的.mat文件
df = scipy.io.loadmat('smoouthedsunspot.mat')
print(df)
#第二，"A"就是上面加载的文件的数据，它以数组形式展现，就是数组名，python中在console中运行1句，就能看到具体的数组名
features = df["A"]
#第三，构造一个表的数据结构，data为表中的数据
dfdata = pd.DataFrame(data=features)
print(dfdata.iloc[0])
dfdata1=dfdata.iloc[0]
#第四，保存为.csv格式的路径;
datapath1 = 'data.csv'
res={}
res["num"]=dfdata1
res=pd.DataFrame(res)
res.to_csv(datapath1, index=False)

