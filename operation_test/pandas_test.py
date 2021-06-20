import numpy as np
import pandas as pd

# 创建DataFrame
df = pd.DataFrame(np.arange(12, 32).reshape((5, 4)), index=["a", "b", "c", "d", "e"], columns=["WW", "XX", "YY", "ZZ"])
xx=np.array(df)
bf=df.columns[2:4]
print(df[bf])

# print(df.mean(0))
# print(df.mean(1))
# print(df["YY"].mean())  # 22.0  指定列的平均值
