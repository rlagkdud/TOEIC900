##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### SECOM DATA - Data Preprocessing 0.3 ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# this project (lecture) is used a scikit-learn librarytyyyyyyyyyyy --> can't USED CUDA...

import pandas as pd 
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


### 원래 데이터를 수집하고, 분류하는 과정을 거쳐야 하지만 (ADP = 데이터 분석 기획),
### 여유가 있는 시간이 아니기에, 기존 과목 (Bigdata_analytics) 에서 사용한 SECOM dataset을 바로 사용합니다.
### 생각해 보면 데이터 수집과 데이터 마이닝은 다르기 때문에 뭐 이렇게 해도 될 것 같기도 하고.

### 이 과정에서는 데이터 전 처리 까지의 과정을 다룹니다. 데이터에 대한 설명은 카톡으로... 
### 예전 처럼 보고서 예쁘게 쓰고 이런건 못하겠고 주석 열심히 달겠습니다.
### 데이터 전처리 과정은 데이터 클리닝이랑 같은 뜻 입니다.
### 이상치, 결측치를 제거하고, 모델이 배우기 쉽게 보정하는 그런 과정. 
### 여기서는 그냥 Step by Step 이라고 쓰고 그냥 전에 쓴 방식 그대로 가져오겠습니다.
### 따로 보고서로 설명도 적을거긴한데 할수있을지는모르겠ㄱ
### 변수 명도 최대한 쉽게 풀어 쓰겠습니다...



# 이 과정은 CSV 파일을 불러온 뒤, 
raw_dataframe = pd.read_csv('./uci-secom.csv')

# SECOM 데이터에 Time이라는 독립변수? 가 있는데 걔는 사실 필요가 없어서 지우고 P/F 결과를 앞으로 가져옵니다.
raw_data_PF = raw_dataframe.loc[:, ['Pass/Fail']]
raw_dataframe = raw_dataframe.drop(['Time'])

# 현재 raw_dataframe의 상태는 Pass/Fail이 앞에 있는 상태입니다.
# print(raw_dataframe)


# 원래 여기에서 Pass 데이터와 Fail 데이터를 나누어 분석하는데
# (결과적으로) 큰 의미가 없어 패스하고, 기술적 통계량도 패스합니다. (print 문 주석 지우면 알수있음
# print(raw_dataframe.describe())


# 이 과정에서 Pass와 Fail 데이터간 불균형을 확인하고, 대책을 세워야 하지만
# 필요없고 Pass 대 Fail 비율이 92 : 8 이었나 94 : 6이었나... 극단적인 상황입니다.
# 이러한 상황에서는 Fail 데이터를 오버샘플링하는 방법이 있는데 그-건 지금 할 단계가 아니니 패스


# 와 생각해보니까 drop 하나 시키고 몇 줄을 그냥 대충 쓰는 기분이긴 한데
# 여기서 상관관계 분석, 특성과 특성 사이의 (각 독립변수간) 연관성을 확인
# 당연히 상관관계가 높으면 Pass/Fail 분류에는 도움이 되지 않으니 제거... (다중공선성이라고 통계용어가있는데 저는 모릅니다)
# 아래 코드는 단순히 모든 피쳐와 피쳐의 상관관계를 계산하여 일정 수치 이상인, 피쳐를 제거합니다.
# 그러므로 아래 코드에 위 df을 넣고 리턴 받아서 뽑아보시면 나옵니다. 

def corrlelation_remove(df, per):
	corr_df = df.corr()
	corr_nan_df = corr_df[corr_df > abs(0.8)]
	t_cols = list(corr_nan_df)
	t_rows = list(corr_nan_df)

	corr_list = []

	# 상관 관계에 대해 확인하고 싶으면 heatmap을 뽑아보면 됩니다.
	for i in range(0, len(t_cols)):
		for j in range(0, len(t_rows)):
			temp = []
			if(corr_nan_df[t_cols[i]][t_rows[j]] > abs(0.8)):
				temp.append(t_cols[i])
				temp.append(t_rows[j])
				temp.append(corr_nan_df[t_cols[i]][t_rows[j]])
				corr_list.append(temp)

	corr_list_df = pd.DataFrame(corr_list)
	corr_result = corr_list_df.drop_duplicates([2], keep="first")

	x = corr_result[0].value_counts()
	xdf = pd.DataFrame(x)
	y = corr_result[1].value_counts()
	ydf = pd.DataFrame(y)

	
	corr_df = pd.concat([xdf, ydf], ignore_index=True, axis=1)
	corr_pc = corr_df.fillna(0)
	corr_df['sum'] = corr_pc[0] + corr_pc[1]
	corr_df = corr_df.sort_values(by=['sum'], axis=0, ascending=False)


	extract = []
	for i in range(0, int(len(corr_df.index) * per)):
		extract.append(list(corr_df.index)[i])
	if("Pass/Fail" in extract):
		extract.remove("Pass/Fail")


	return df.drop(extract, axis=1)

###

# 졸리니 여기까지만 설명하겠습니다... 
# 아래 코드는 결측치에 대한 내용인데,
# 결측치란 말 그대로 측정이 되지 않은 NaN 값이므로
# 그 값의 비율이 높으면? --> 피쳐 제거
# 어느정도 괜찮은 느낌이면? --> IterativeImputer라는 Scikit-Learn의 좋은 Imputation 도구가 이썽서 사요ㅕㅇ
# 근데 max-iter는 기본이 10이고, 높을수록 아웃풋이 막 뛰어나거나 하지는 않아서 (SECOM 데이터 한정)
# 그래서 적당히 max_iter 한 8이나 16정도로 줄이면 될거같은 느낌

def missing_value_processing(df):

	null_data = df.isnull().sum()
	null_df = pd.DataFrame(null_data)
	null_df['null_per'] = (null_df[0] / len(null_df.index))

	null_list = null_df[null_df['null_per'] > 0.6].index
	remove_data = df.drop(null_list, axis=1)
	save_cols = list(remove_data.describe().columns)

	try:
		save_cols = list(remove_data.describe().columns)
		save_char_df = remove_data.loc[:, ['Time', 'Pass/Fail']]
		imp_data = remove_data.drop(['Time', "Pass/Fail"], axis=1)
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=30, verbose=False).fit_transform(imp_data), columns=save_cols[1:])
		processed_df = pd.concat([save_char_df, imputed_df], axis=1)
	except:
		save_cols = list(remove_data.describe().columns)
		imputed_df = pd.DataFrame(IterativeImputer(max_iter=30, verbose=False).fit_transform(remove_data), columns=save_cols)
		processed_df = imputed_df
	return processed_df
