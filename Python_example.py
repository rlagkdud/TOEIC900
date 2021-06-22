##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### SECOM DATA - Data Preprocessing 0.3 ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# 근-본 라이브러리
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 결측치 대체를 위한 라이브러리
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 간단히 과정의 흐름을 보기 위해 만들었기 때문에 모듈화 X
# 이해 안되는 부분은 코멘트 남겨주세요... 



# secom data를 불러와서 크게 중요하지 않은 변수인 시간을 제거하고, Pass/Fail을 예측하는 것이
# 이 프로젝트의 목표이자 Y_label 같은 그런거라 자주 쓰이므로 맨 앞으로 데려오고
# 그 다음에는 각 Feature에 구분하기 편하게 앞에 F라는 프리픽스를 붙여줌
raw_dataframe = pd.read_csv('./uci-secom.csv')
raw_data_PF = raw_dataframe.loc[:, ['Pass/Fail']]
raw_dataframe = raw_dataframe.drop(['Time', 'Pass/Fail'], axis=1).add_prefix('F')


# 이 데이터셋의 기술적 통계량을 확인하기 위해 Pass/Fail 데이터 분리
df_secom = pd.concat([raw_data_PF, raw_dataframe], axis=1)
df_secom_pass = df_secom[df_secom['Pass/Fail'] == -1]
df_secom_fail = df_secom[df_secom['Pass/Fail'] == 1]

##print("PASS - " + str(len(df_secom_pass.index)))
##print("FAIL - " + str(len(df_secom_fail.index)))
# [출력결과] PASS / FAIL이 1463 : 104으로 매우 편향된 좋지 않은 데이터임을 알 수 있음
# [해결방안] 최대한 성능에 도움이 안되는 특성을 제거하고, 오버샘플링 진행



# 이번엔 데이터의 결측 비율을 확인... 
secom_null = df_secom.isnull().sum()
df_secom_null = pd.DataFrame(secom_null).drop(['Pass/Fail'], axis=0)
df_secom_null['null_per'] = (df_secom_null[0] / len(df_secom_null.index))
null_list = df_secom_null[df_secom_null['null_per'] > 0.6].index
##print(null_list)
# [출력결과] 결측치가 60% 이상인 특성들을 확인 : 32 Features
# [해결방안] 일반적으로 45~55% 이상은 제거하기 때문에 60% 이상인 특성은 제거
# [주의] 원본에서 Drop하므로 유의하며 진행



# 다중대체법의 장점은 결측치 비율이나 변수의 개수에 크게 영향을 받지 않으며
# 대체하는 값에 대한 불확실성에 대한 정보를 사용할 수 있다.
# 그러나 다중 대체법은 매번 같은 값을 제공하지 않는다. 무작위성이 있음. 

df_secom = df_secom.drop(null_list, axis=1)
##print(df_secom)
# [출력결과] 전체 피쳐 수 558개로 줄어듬.
# [다음과정] 이제 결측 데이터를 다중 대체법을 통해 채워 나간다...
# [레퍼런스] https://www.jpmph.org/upload/pdf/jpmph-37-3-209.pdf
#           서울대학교 통계학과 특강원고_윤성철



secom_cols = list(df_secom.describe().columns)
secom_pf = df_secom.loc[:, ['Pass/Fail']]
df_secom = df_secom.drop(['Pass/Fail'], axis=1)
impute_secom = pd.DataFrame(IterativeImputer(max_iter=3, verbose=False).fit_transform(df_secom), columns=secom_cols[1:])
df_secom = pd.concat([secom_pf, impute_secom], axis=1)
print(df_secom)
# [출력결과] warnings.warn("[IterativeImputer] Early stopping criterion not"
#           Ealy Stoping 에러가 뜨는 이유는 max_iter가 되기 전에 이미 다 끝났다는 이야기라 max_iter를 줄여줍니다.
# [다음과정] 이제 결측치도 채웠겠다 다시 한번 기술적 통계를 시각화 해서 봅시다.
# [레퍼런스] https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

'''
## 이건 실제 출력물 ##
C:\\Python\\lib\\site-packages\\sklearn\\impute\\_iterative.py:685: ConvergenceWarning: [IterativeImputer] Early stopping criterion not reached.
  warnings.warn("[IterativeImputer] Early stopping criterion not"
      Pass/Fail       F0       F1  ...      F587      F588        F589
0            -1  3030.93  2564.00  ...  0.015081  0.004993  100.574459
1            -1  3095.78  2465.14  ...  0.020100  0.006000  208.204500
2             1  2932.61  2559.94  ...  0.048400  0.014800   82.860200
3            -1  2988.72  2479.90  ...  0.014900  0.004400   73.843200
4            -1  3032.24  2502.87  ...  0.014900  0.004400   73.843200
...         ...      ...      ...  ...       ...       ...         ...
1562         -1  2899.41  2464.36  ...  0.013800  0.004700  203.172000
1563         -1  3052.31  2522.55  ...  0.013800  0.004700  203.172000
1564         -1  2978.81  2379.78  ...  0.008600  0.002500   43.523100
1565         -1  2894.92  2532.01  ...  0.024500  0.007500   93.494100
1566         -1  2944.92  2450.76  ...  0.016200  0.004500  137.784400
'''
# imputation 과정은 생각보다 시간이 많이 걸립니다. Scikit-Learn은 CUDA연산도 지원안하고 ㅠㅠ
# 실제 구현 시에는 이 내용을 to_csv 를 이용해 저장해놓고 씁니다. 
# 대강대강 히트맵이랑 박스플랏 정도 하나씩 봅시다.


# 아래는 특성간 상관관계 히트맵을 출력하는 내용인데, 대칭이라 나머지 반을 지우고 싶으면
# numpy를 이용해서 지울 부분에 대한 mask를 만들어 heatmap의 mask 파라미터에 넣어주면 됩니다.
corr = df_secom.corr()
sns.heatmap(corr, cmap='Greens', annot=False, vmin=0, vmax=1)
plt.show()
##이 결과물은 figure_1.png으로 저장되었습니다.
