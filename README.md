# binary-classification

## 위스콘신 유방암 데이터셋 - 이진분류(Binary Classification) 문제
- Feature : 종양에 대한 다양한 측정값들
- Target의 class : 0 - malignant(악성종양), 1 - benign(양성종양)

## 1. import 
```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
## 2. dataset, dataloader 생성
### 데이터 생성
```python
X, y = load_breast_cancer(return_X_y = True) # 인풋, 아웃풋 데이터 ndarray로 준다
print(type(X), type(y))
print(X.shape, y.shape)
print(np.unique(y))

<class 'numpy.ndarray'> <class 'numpy.ndarray'>
(569, 30) (569,)
[0 1]
```
```python
# y shape을 2차원으로 변경 -> 모델 출력 shape과 맞춰주기 위함
# (batch_size, 1)
y = y.reshape(-1, 1)
y.shape

(569, 1)
```
```
### train test set 분리
```python
# train / test set 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, # 나눌 대상 
                                                    test_size=0.25, # 나눌 비율
                                                    stratify=y # class 별 비율을 맞춰서 나눔
                                                   )

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

### 데이터 전처리
```python
# 전처리 - feature scaling (컬럼들의 scale(척도)를 맞춘다)
# standardscaler => 평균: 0, 표준편차: 1을 기준으로 맞춘다
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 위에 두개 합친거
X_test_scaled = scaler.transform(X_test) # trainset 으로 fit한 scaler로 변환
```
```python
# ndarray -> Tensor로 변환 -> dataset 구성 -> dataloader 구성
# ndarray -> torch.Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
```
### dataset 생성
```python
# dataset 생성 -> 메모리의 tensor를 dataset으로 생성 -> tensordataset
trainset = TensorDataset(X_train_tensor, y_train_tensor)
testset = TensorDataset(X_test_tensor, y_test_tensor)
```
### dataloader 생성
```python
# DataLoader
train_loader = DataLoader(trainset, batch_size = 200, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size=len(testset))
```

## 3. 모델 정의
```python
class BCModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lr1 = nn.Linear(30,32) # input은 X_train의 shape의 열부분
        self.lr2 = nn.Linear(32,8)
        self.lr3 = nn.Linear(8, 1) # 이진분류이기 때문에 positive output 하나
        

    def forward(self, X):
        # flatten 할 필요없다 데이터 1차원이기 때문에
        # out = self.lr1(X) 
        # out = nn.ReLU(out)
        out = nn.ReLU()(self.lr1(X)) # 위에 두개 합친거
        out = nn.ReLU()(self.lr2(out))
        # 이진분류 출력값 처리 -> linear()는 한개의 값을 출력 -> 확률값으로 변경 -> sigmoid/logistic 함수를 activation함수로 사용
        out = self.lr3(out)
        out = nn.Sigmoid()(out)
        return out
```





