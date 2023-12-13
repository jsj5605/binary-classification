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
## 4. 학습(train)
```python
import time
## 하이퍼파라미터
LR = 0.001
N_EPOCH = 1000

# 모델 생성
model = BCModel().to(device)
# loss 함수
loss_fn = nn.BCELoss() # binary cross entropy loss
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

s = time.time()
######################### 
# 에폭별 검증 -> train loss, validation loss, validation accuracy
# 조기종료(Early Stopping) - 성능 개선이 안되면 학습을 중단
# 가장 좋은 성능을 내는 에폭의 모델을 저장. 
#  조기종료/모델 저장 ==> validation loss 기준.
######################### 

## 결과 저장할 리스트
train_loss_list, valid_loss_list, valid_acc_list = [], [], []

### 모델 저장, 조기종료 관련 변수
best_score = torch.inf  # valid loss
save_bcmodel_path = "models/bc_best_model.pth"

patience = 20  # 성능이 개선 될때 까지 몇 에폭 기다릴 것인지.
trigger_cnt = 0 # 성능이 개선 될때 까지 현재 몇번째 기다렸는지.


for epoch in range(N_EPOCH):
    
    # train
    model.train()
    train_loss = 0.0
    for X_train, y_train in train_loader:
        # 한 step
        X_train, y_train = X_train.to(device), y_train.to(device)
        pred_train = model(X_train) # 예측
        loss = loss_fn(pred_train, y_train) # 오차계산
        # 파라미터 업데이트
        optimizer.zero_grad() # 초기화
        loss.backward() # grad 계산
        optimizer.step() # 파라미터 update
        
        train_loss += loss.item()
    train_loss /= len(train_loader) # 현재 epoch의 평균 train loss 계산
    
    #### 검증(validation)
    model.eval()
    valid_loss, valid_acc = 0.0, 0.0
    with torch.no_grad():
        for X_valid, y_valid in test_loader:
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)
            pred_valid = model(X_valid) # 값: 1개 - positive일 확률 ==> Loss 계산
            pred_label = (pred_valid > 0.5).type(torch.int32)  # label ==> 정확도 계산
            
            # loss
            loss_valid = loss_fn(pred_valid, y_valid)
            valid_loss += loss_valid.item()
            # 정확도
            valid_acc += torch.sum(pred_label == y_valid).item()
    # valid 검증 결과 계산
    valid_loss /= len(test_loader)
    valid_acc /= len(test_loader.dataset)
    
    print(f"[{epoch+1}/{N_EPOCH}] train loss: {train_loss}, valid loss: {valid_loss}, valid accuracy: {valid_acc}")
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_acc)
    
    ##### 모델 저장 및 조기종료 처리
    if valid_loss < best_score: # 성능 개선
        print(f"===> {epoch+1}에폭에서 모델 저장. 이전 score: {best_score}, 현재 score: {valid_loss}")
        torch.save(model, save_bcmodel_path)
        best_score = valid_loss
        trigger_cnt = 0
    else: # 성능개선이 안됨.
        trigger_cnt += 1
        if patience == trigger_cnt: # 조기종료
            print(f"######## Early Stop: {epoch+1}")
            break
    
    
e = time.time()
print(f"학습시간: {e-s}초")
```
## 5. 학습 시각화
```python
plt.plot(train_loss_list, label="train")
plt.plot(valid_loss_list, label="validation")
plt.ylim(0, 0.1)
plt.legend()
plt.show()
```
![dfd](https://github.com/jsj5605/binary-classification/assets/141815934/803df189-24db-492f-8072-ad0eef39b1fb)

## 6. 모델 평가
```python
best_model = torch.load(save_bcmodel_path)
pred_new = best_model(X_test_tensor)
pred_new.shape

pred_new[:5]

output:
tensor([[9.9998e-01],
        [9.9998e-01],
        [1.0429e-05],
        [9.9850e-01],
        [9.9959e-01]], grad_fn=<SliceBackward0>)

# 확률->class index
pred_new_label = (pred_new > 0.5).type(torch.int32)
pred_new_label[:5]

output:
tensor([[1],
        [1],
        [0],
        [1],
        [1]], dtype=torch.int32)

y_test_tensor[:5]

output:
tensor([[1.],
        [1.],
        [0.],
        [1.],
        [1.]])
```





