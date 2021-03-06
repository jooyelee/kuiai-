#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout


# 

# In[58]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


labeled_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/molding/labeled_data.csv")


# In[5]:


labeled_data.head()


# In[6]:


labeled_data.describe()


# # 데이터 전처리

# ##### 데이터 유일성 --> 0.65로 만족 안함

# In[7]:



a = len(labeled_data["_id"].unique())
print(a/len(labeled_data))


# In[8]:


result = labeled_data.drop_duplicates()


# In[9]:


a = len(result["_id"].unique())
print(a/len(result))


# In[10]:


labeled_data = result


# #### 데이터 완전성 --> 100%
# 

# In[11]:


labeled_data.isna().sum()


# ##### 데이터 유효성: 유효정보에 대한 접근이 없어서 시간 데이터 형식만 바꾼다

# In[12]:


labeled_data['TimeStamp'] = pd.to_datetime(labeled_data['TimeStamp'],format='%Y-%m-%dT%H:&M:SZ',errors="coerce")
labeled_data['PART_FACT_PLAN_DATE'] = pd.to_datetime(labeled_data['PART_FACT_PLAN_DATE'],format='%Y-%m-%d 오전 12:00:00',errors="coerce")


# #### 데이터 정확성 : PassOrFail & reason과의 상관관계 보기

# In[13]:


labeled_data["Reason"].unique()


# In[14]:


labeled_data["PassOrFail"].unique()


# In[15]:


a = labeled_data['Reason'] == "None"
b = labeled_data['PassOrFail'] == "Y" # 정상이라면 불량 난 이유를 굳이 써줄 이유 없음
# 정상 == 정상?
print(len(labeled_data[a])-len(labeled_data[b]))


# In[16]:


a = labeled_data['Reason'] != "None"
b = labeled_data['PassOrFail'] != "Y" # 정상이라면 불량 난 이유를 굳이 써줄 이유 없음
# 비정상 == 비정상?
print(len(labeled_data[a])-len(labeled_data[b]))


# # 데이터 selection

# In[17]:


labeled_data.corr()


# In[18]:


labeled_data["EQUIP_NAME"].value_counts()


# In[19]:


labeled_data["PART_NAME"].value_counts()


# 지울 컬럼들
# 
# --> 무의미한 STR값들 + 시간데이터는 지우겠음
# _id                        
# TimeStamp                   
# PART_FACT_PLAN_DATE         
# PART_FACT_SERIAL            
# PART_NAME                   
# EQUIP_CD                    
# EQUIP_NAME                        
# Reason                      
# --> 값이 0인 COLUMN들
# Mold_Temperature_1          
# Mold_Temperature_2          
#          
# Mold_Temperature_5          
# Mold_Temperature_6          
# Mold_Temperature_7          
# Mold_Temperature_8          
# Mold_Temperature_9          
# Mold_Temperature_10         
# Mold_Temperature_11         
# Mold_Temperature_12        

# In[20]:


def delete_column(data, machine_name,product_name):
  machine_ = data["EQUIP_NAME"] == machine_name
  product_ = data["PART_NAME"] == product_name
  data = data[machine_ & product_]

  data.drop(['_id', 'TimeStamp', 'PART_FACT_PLAN_DATE', 'PART_FACT_SERIAL',
       'PART_NAME', 'EQUIP_CD', 'EQUIP_NAME', 'Reason',
       'Mold_Temperature_1', 'Mold_Temperature_2', 'Mold_Temperature_5', 'Mold_Temperature_6',
       'Mold_Temperature_7', 'Mold_Temperature_8', 'Mold_Temperature_9',
       'Mold_Temperature_10', 'Mold_Temperature_11', 'Mold_Temperature_12'],
       axis=1,inplace=True)
  return data


# In[21]:


machine_name ="650톤-우진2호기"
product_name = ["CN7 W/S SIDE MLD'G RH","CN7 W/S SIDE MLD'G LH","RG3 MOLD'G W/SHLD, RH","RG3 MOLD'G W/SHLD, LH " ]

cn7_rh = delete_column(labeled_data, machine_name,product_name[0])
cn7_lh = delete_column(labeled_data, machine_name,product_name[1])
rg3_rh = delete_column(labeled_data, machine_name,product_name[2])
rg3_lh = delete_column(labeled_data, machine_name,product_name[3])


# In[22]:


cn7_rh.head()


# In[23]:


cn7 = pd.concat([cn7_lh,cn7_rh])
rg3 = pd.concat([rg3_lh,rg3_rh])


# pass or fail 을 각각 0 ,1 으로 바꾸기

# In[24]:


cn7["PassOrFail"] = cn7["PassOrFail"].replace("Y",1).replace("N",0)
rg3["PassOrFail"] = rg3["PassOrFail"].replace("Y",1).replace("N",0)


# In[25]:


cn7.head()


# In[26]:


rg3.head()


# In[ ]:


plt.figure(figsize = (30,30))
for index,value in enumerate(cn7):
  sub = plt.subplot(6,5,index+1)
  sub.hist(cn7[value],linewidth=3)
  plt.title(value)


# In[28]:


cn7['Switch_Over_Position'].unique()


# In[29]:


cn7['Barrel_Temperature_7'].unique()


# Switch_Over_Position
# Barrel_Temperature_7
# 의 값이 모두 unique함으로 cn7에서 위 두 칼럽을 지운다

# In[30]:


cn7.drop(["Switch_Over_Position","Barrel_Temperature_7"],axis=1,inplace=True)


# In[31]:


cn7.describe()


# In[ ]:


plt.figure(figsize = (30,30))
for index,value in enumerate(rg3):
  sub = plt.subplot(6,5,index+1)
  sub.hist(rg3[value],linewidth=3)
  plt.title(value)


# In[33]:


rg3['Switch_Over_Position'].unique()


# In[34]:


rg3["Clamp_Open_Position"].unique()


# In[35]:


rg3["Clamp_Open_Position"].describe()


# In[36]:


rg3['Barrel_Temperature_7'].unique()


# In[37]:


rg3.drop(["Switch_Over_Position","Clamp_Open_Position","Barrel_Temperature_7"],axis=1,inplace=True)


# In[38]:


rg3.head()


# In[59]:


# 불량 정상 갯수 확인

cn7_Y = cn7[cn7["PassOrFail"]==1]
cn7_N = cn7[cn7["PassOrFail"]==0]

print(f"cn7의 정상품 수 {len(cn7_Y)}  cn7의 불량품 수 {len(cn7_N)}")

rg3_Y = rg3[rg3["PassOrFail"]==1]
rg3_N = rg3[rg3["PassOrFail"]==0]

print(f"rg3의 정상품 수 {len(rg3_Y)}  rg3의 불량품 수 {len(rg3_N)}")


# In[40]:


cn7.columns


# In[41]:


rg3.columns


# In[42]:


plt.figure(figsize = (30,30))
for index,value in enumerate(cn7_Y):
  sub = plt.subplot(6,5,index+1)
  sub.hist(cn7_Y[value],linewidth=3)
  plt.title(value)


# In[43]:


plt.figure(figsize = (30,30))
for index,value in enumerate(rg3_Y):
  sub = plt.subplot(6,5,index+1)
  sub.hist(rg3_Y[value],linewidth=3)
  plt.title(value)


# In[60]:


# y 종속변수 제거

cn7_Y_y = cn7_Y["PassOrFail"]
cn7_N_y = cn7_N["PassOrFail"]
cn7_Y_x = cn7_Y.iloc[:,1:]
cn7_N_x = cn7_N.iloc[:,1:]

rg3_Y_y = rg3_Y["PassOrFail"]
rg3_N_y = rg3_N["PassOrFail"]
rg3_Y_x = rg3_Y.iloc[:,1:]
rg3_N_x = rg3_N.iloc[:,1:]


# In[61]:


scaler = MinMaxScaler()

cn7_Y = scaler.fit_transform(cn7_Y_x)
cn7_N = scaler.fit_transform(cn7_N_x)

rg3_Y =  scaler.fit_transform(rg3_Y_x)
rg3_N =  scaler.fit_transform(rg3_N_x)


# In[ ]:


rg3_N


# training data
# test data 

# In[47]:


rg3_Y_x.shape


# In[69]:


rg3_N_x.shape


# In[48]:


cn7_Y_x.shape


# In[49]:


cn7_N_x.shape


# In[ ]:





# In[62]:


cn7_train = cn7_Y[:2400] # 아직 결과 데이터 안지운 상태 (둘다)
cn7_test_y = cn7_Y[2400:]
cn7_test_n =cn7_N


# In[63]:


rg3_train = rg3_Y[:380]
rg3_test_y = rg3_Y[380:]
rg3_test_n = rg3_N


# In[52]:


'''rg3_test_y = pd.DataFrame(rg3_test_y)
rg3_test_y.to_csv("rg3_test_y.csv")

rg3_test_n = pd.DataFrame(rg3_test_n)
rg3_test_n.to_csv("rg3_test_n.csv")

cn7_test_y = pd.DataFrame(cn7_test_y)
cn7_test_y.to_csv("cn7_test_y.csv")

cn7_test_n = pd.DataFrame(cn7_test_n)
cn7_test_n.to_csv("cn7_test_n.csv")
rg3_train = pd.DataFrame(rg3_train)
rg3_train.to_csv("rg3_train.csv")
cn7_train = pd.DataFrame(cn7_train)
cn7_train.to_csv("cn7_train.csv")'''


# In[64]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping


# # cn7

# In[54]:


# 20(relu) - 10(relu) - 20(relu) - ouput(sigmoid) + early stopping
def AE(epoch, batch):
  # 인코더
  dropout_encoder = Sequential([
                                
      Dropout(0.3),
      Dense(20,activation="relu"), 
      Dense(10,activation="relu")])
  # 디코더
  dropout_decoder = Sequential([
      Dense(20,activation="relu",input_shape=[10]),
      Dense(cn7_train.shape[1],activation="sigmoid")])
  dropout_AE = Sequential([dropout_encoder,dropout_decoder])

  # 손실함수 옵티마이저 정의
  dropout_AE.compile(loss="mse",optimizer=Adam(lr=0.03),metrics=["accuracy"])
  # 모델훈련
  history = dropout_AE.fit(cn7_train,cn7_train,batch_size=batch,epochs=epoch,validation_split=0.2,callbacks=[EarlyStopping(monitor="val_loss",patience=10,mode="min")])
  plt.figure(1)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["accuracy"],label="Training Acc")
  plt.plot(history.history["val_accuracy"],label="Validation Acc")
  plt.legend()
  plt.figure(2)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["loss"],label="Training Loss")
  plt.plot(history.history["val_loss"],label="Validation Loss")
  plt.legend()
  plt.show()

  cn7_train_pred = dropout_AE.predict(cn7_train)
  cn7_train_loss = np.mean(np.square(cn7_train_pred - cn7_train),axis=1)
  threshold = np.mean(cn7_train_loss)+1.96*np.std(cn7_train_loss)
  print(f"복원 오류 임계치: {threshold}") # 뭔가 이상
  print("-"*60)
  # 예측값

  # 평가 데이터 정상
  cn7_predict_y = dropout_AE.predict(cn7_test_y)
  cn7_test_y_mse = np.mean(np.square(cn7_predict_y - cn7_test_y),axis=1)
  # 시각화
  plt.figure(3)
  plt.hist(cn7_test_y_mse,bins = 30)
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.title("testing normal data")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_y_anomalies = len(cn7_test_y_mse[cn7_test_y_mse > threshold])
  fn = cn7_test_y_anomalies
  # 정상을 정상으로 판단
  tp = len(cn7_test_y_mse)-fn
  print(f"불량/정상 갯수 : {fn}")
  print(f"정상/정상 갯수 : {tp}")
  
 # 평가 데이터 불량

  cn7_predict_n = dropout_AE.predict(cn7_test_n)
  cn7_test_n_mse = np.mean(np.square(cn7_predict_n - cn7_test_n),axis=1)
  # 시각화
  plt.figure(4)
  plt.hist(cn7_test_n_mse,bins = 30)
  plt.title("testing abnormal data")
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_n_anomalies = len(cn7_test_n_mse[cn7_test_n_mse > threshold])
  tn = cn7_test_n_anomalies
  # 실제 불량인데 정상으로 판단한 데이터
  fp = len(cn7_test_n_mse)-tn
  print(f"불량/불량 갯수 : {tn}")
  print(f"정상/불량 갯수 : {fp}")
  return (epoch,batch,fn,tp,tn,fp)
  #df_new = pd.DataFrame([(epoch,batch,fn,tp,tn,fp)])
  #relu_201020=relu_201020.append(df_new,ignore_index=True)

  '''cn7_true = np.concatenate([ np.zeros(len(cn7_test_y_anomalies)), np.ones(len(cv7_test_n_anomalies))])
  cn7_prediction = np.concatenate([cv7_test_y_anomalies,cv7_test_n_anomalies])

  from sklearn.metrics import confusion_matrix
  print(confusion_matrix(cn7_true,cn7_prediction))'''


# In[ ]:


def allsummery(epochs,batches,iteration):
  arr = []
  for i in epochs:
    for j in batches:
      for _ in range(iteration):
          t = AE(i,j)
          arr.append(t)
  df = pd.DataFrame(arr,columns=["epoch","batch","fn","tp","tn","fp"])
  return df
relu_201020 = allsummery([500,100,50,30],[30,20,10],30)


# In[ ]:


relu_201020


# In[ ]:


relu_201020_2 = allsummery([40,50,60],[25,28,30,32,35])


# In[ ]:


relu_201020_2


# In[ ]:





# In[66]:


# 20(relu) - 10(relu) - 20(relu) - ouput(sigmoid) + no early stopping


def AE_nostop(epoch, batch):
  # 인코더
  dropout_encoder = Sequential([
                                
      Dropout(0.3),
      Dense(20,activation="relu"), 
      Dense(10,activation="relu")])
  # 디코더
  dropout_decoder = Sequential([
      Dense(20,activation="relu",input_shape=[10]),
      Dense(cn7_train.shape[1],activation="sigmoid")])
  dropout_AE = Sequential([dropout_encoder,dropout_decoder])

  # 손실함수 옵티마이저 정의
  dropout_AE.compile(loss="mse",optimizer=Adam(lr=0.03),metrics=["accuracy"])
  # 모델훈련
  history = dropout_AE.fit(cn7_train,cn7_train,batch_size=batch,epochs=epoch,validation_split=0.2)
  plt.figure(1)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["accuracy"],label="Training Acc")
  plt.plot(history.history["val_accuracy"],label="Validation Acc")
  plt.legend()
  plt.figure(2)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["loss"],label="Training Loss")
  plt.plot(history.history["val_loss"],label="Validation Loss")
  plt.legend()
  plt.show()

  cn7_train_pred = dropout_AE.predict(cn7_train)
  cn7_train_loss = np.mean(np.square(cn7_train_pred - cn7_train),axis=1)
  threshold = np.mean(cn7_train_loss)+1.96*np.std(cn7_train_loss)
  print(f"복원 오류 임계치: {threshold}") # 뭔가 이상
  print("-"*60)
  # 예측값

  # 평가 데이터 정상
  cn7_predict_y = dropout_AE.predict(cn7_test_y)
  cn7_test_y_mse = np.mean(np.square(cn7_predict_y - cn7_test_y),axis=1)
  # 시각화
  plt.figure(3)
  plt.hist(cn7_test_y_mse,bins = 30)
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.title("testing normal data")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_y_anomalies = len(cn7_test_y_mse[cn7_test_y_mse > threshold])
  fn = cn7_test_y_anomalies
  # 정상을 정상으로 판단
  tp = len(cn7_test_y_mse)-fn
  print(f"불량/정상 갯수 : {fn}")
  print(f"정상/정상 갯수 : {tp}")
  
 # 평가 데이터 불량

  cn7_predict_n = dropout_AE.predict(cn7_test_n)
  cn7_test_n_mse = np.mean(np.square(cn7_predict_n - cn7_test_n),axis=1)
  # 시각화
  plt.figure(4)
  plt.hist(cn7_test_n_mse,bins = 30)
  plt.title("testing abnormal data")
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_n_anomalies = len(cn7_test_n_mse[cn7_test_n_mse > threshold])
  tn = cn7_test_n_anomalies
  # 실제 불량인데 정상으로 판단한 데이터
  fp = len(cn7_test_n_mse)-tn
  print(f"불량/불량 갯수 : {tn}")
  print(f"정상/불량 갯수 : {fp}")
  return (epoch,batch,fn,tp,tn,fp)
  #df_new = pd.DataFrame([(epoch,batch,fn,tp,tn,fp)])
  #relu_201020=relu_201020.append(df_new,ignore_index=True)

  '''cn7_true = np.concatenate([ np.zeros(len(cn7_test_y_anomalies)), np.ones(len(cv7_test_n_anomalies))])
  cn7_prediction = np.concatenate([cv7_test_y_anomalies,cv7_test_n_anomalies])

  from sklearn.metrics import confusion_matrix
  print(confusion_matrix(cn7_true,cn7_prediction))'''
def allsummery(epochs,batches,iter):
  arr = []
  for i in epochs:
    for j in batches:
      for k in range(iter):
        print("-"*60)
        print(f"{k}th iteration epoch {i} batch {j}")
        t = AE_nostop(i,j)
        arr.append(t)
  df = pd.DataFrame(arr,columns=["epoch","batch","fn","tp","tn","fp"])
  return df
relu_201020_nostop = allsummery([100,50,30],[30,20,10],30)


# In[67]:


relu_201020_2 = relu_201020_nostop
relu_201020_2["precision"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fp"])
relu_201020_2["recall"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fn"])
relu_201020_2["accuracy"] = (relu_201020_2["tp"]+relu_201020_2["tn"])/(relu_201020_2["tp"]+relu_201020_2["fp"]+relu_201020_2["fn"]+relu_201020_2["tn"])
relu_201020_2["F1"] = 2*relu_201020_2["recall"]*relu_201020_2["precision"]/(relu_201020_2["recall"]+relu_201020_2["precision"])
relu_201020_2.head()


# In[68]:


relu_201020_2.groupby(["epoch","batch"]).mean()


# In[90]:


rg3_test_n.shape


# In[91]:


rg3_test_y.shape


# # rg3
# 

# In[92]:


rg3_train = rg3_Y[:380]
rg3_test_y = rg3_Y[380:]
rg3_test_n = rg3_N

def AE(epoch, batch):
  # 인코더
  dropout_encoder = Sequential([
                                
      Dropout(0.1),
      Dense(15,activation="tanh"), 
      ]) #Dense(6,activation="relu")
  # 디코더
  dropout_decoder = Sequential([

      Dense(rg3_train.shape[1],activation="sigmoid")])
  dropout_AE = Sequential([dropout_encoder,dropout_decoder])

  # 손실함수 옵티마이저 정의
  dropout_AE.compile(loss="mse",optimizer=Adam(lr=0.03),metrics=["accuracy"])
  # 모델훈련
  history = dropout_AE.fit(rg3_train,rg3_train,batch_size=batch,epochs=epoch,validation_split=0.2,callbacks=[EarlyStopping(monitor="val_loss",patience=10,mode="min")])
  plt.figure(1)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["accuracy"],label="Training Acc")
  plt.plot(history.history["val_accuracy"],label="Validation Acc")
  plt.legend()
  plt.figure(2)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["loss"],label="Training Loss")
  plt.plot(history.history["val_loss"],label="Validation Loss")
  plt.legend()
  plt.show()

  rg3_train_pred = dropout_AE.predict(rg3_train)
  rg3_train_loss = np.mean(np.square(rg3_train_pred - rg3_train),axis=1)
  threshold = np.mean(rg3_train_loss)+1.96*np.std(rg3_train_loss)
  print(f"복원 오류 임계치: {threshold}") # 뭔가 이상
  print("-"*60)
  # 예측값

  # 평가 데이터 정상
  rg3_predict_y = dropout_AE.predict(rg3_test_y)
  rg3_test_y_mse = np.mean(np.square(rg3_predict_y - rg3_test_y),axis=1)
  # 시각화
  plt.figure(3)
  plt.hist(rg3_test_y_mse,bins = 30)
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.title("testing normal data")
  plt.show()

  #불량으로 판단한 데이터 확인
  rg3_test_y_anomalies = len(rg3_test_y_mse[rg3_test_y_mse > threshold])
  fn = rg3_test_y_anomalies
  # 정상을 정상으로 판단
  tp = len(rg3_test_y_mse)-fn
  print(f"불량/정상 갯수 : {fn}")
  print(f"정상/정상 갯수 : {tp}")
  
 # 평가 데이터 불량

  rg3_predict_n = dropout_AE.predict(rg3_test_n)
  rg3_test_n_mse = np.mean(np.square(rg3_predict_n - rg3_test_n),axis=1)
  # 시각화
  plt.figure(4)
  plt.hist(rg3_test_n_mse,bins = 30)
  plt.title("testing abnormal data")
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.show()

  #불량으로 판단한 데이터 확인
  rg3_test_n_anomalies = len(rg3_test_n_mse[rg3_test_n_mse > threshold])
  tn = rg3_test_n_anomalies
  # 실제 불량인데 정상으로 판단한 데이터
  fp = len(rg3_test_n_mse)-tn
  print(f"불량/불량 갯수 : {tn}")
  print(f"정상/불량 갯수 : {fp}")
  return (epoch,batch,fn,tp,tn,fp)
  #df_new = pd.DataFrame([(epoch,batch,fn,tp,tn,fp)])
  #relu_201020=relu_201020.append(df_new,ignore_index=True)

def allsummery(epochs,batches,iter):
  import time
  time_elapsed = []
  arr = []
  for i in epochs:
    for j in batches:
      for k in range(iter):
        print("-"*60)
        print(f"{k}th iteration epoch {i} batch {j}")
        start = time.time()
        t = AE(i,j)
        end = time.time()
        time_elapsed.append(end-start)
        arr.append(t)
  df = pd.DataFrame(arr,columns=["epoch","batch","fn","tp","tn","fp"])
  df["elapsed time"] = time_elapsed
  return df
relu_201020_rg3 = allsummery([100,40,40],[20, 15,10],1)
relu_201020_2 = relu_201020_rg3
relu_201020_2["precision"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fp"])
relu_201020_2["recall"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fn"])
relu_201020_2["accuracy"] = (relu_201020_2["tp"]+relu_201020_2["tn"])/(relu_201020_2["tp"]+relu_201020_2["fp"]+relu_201020_2["fn"]+relu_201020_2["tn"])
relu_201020_2["F1"] = 2*relu_201020_2["recall"]*relu_201020_2["precision"]/(relu_201020_2["recall"]+relu_201020_2["precision"])
relu_201020_2.head()


# In[85]:


relu_201020_2.groupby(["epoch","batch"]).mean()


# ## split 비율 = 6:4 vs 7:3

# In[ ]:


# 6:4 
cn7_train = cn7_Y[:2400] # 아직 결과 데이터 안지운 상태 (둘다)
cn7_test_y = cn7_Y[2400:]
cn7_test_n =cn7_N

def AE(epoch, batch):
  # 인코더
  dropout_encoder = Sequential([
                                
      Dropout(0.3),
      Dense(20,activation="relu"), 
      Dense(10,activation="relu")])
  # 디코더
  dropout_decoder = Sequential([
      Dense(20,activation="relu",input_shape=[10]),
      Dense(cn7_train.shape[1],activation="sigmoid")])
  dropout_AE = Sequential([dropout_encoder,dropout_decoder])

  # 손실함수 옵티마이저 정의
  dropout_AE.compile(loss="mse",optimizer=Adam(lr=0.03),metrics=["accuracy"])
  # 모델훈련
  history = dropout_AE.fit(cn7_train,cn7_train,batch_size=batch,epochs=epoch,validation_split=0.2,callbacks=[EarlyStopping(monitor="val_loss",patience=10,mode="min")])
  plt.figure(1)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["accuracy"],label="Training Acc")
  plt.plot(history.history["val_accuracy"],label="Validation Acc")
  plt.legend()
  plt.figure(2)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["loss"],label="Training Loss")
  plt.plot(history.history["val_loss"],label="Validation Loss")
  plt.legend()
  plt.show()

  cn7_train_pred = dropout_AE.predict(cn7_train)
  cn7_train_loss = np.mean(np.square(cn7_train_pred - cn7_train),axis=1)
  threshold = np.mean(cn7_train_loss)+1.96*np.std(cn7_train_loss)
  print(f"복원 오류 임계치: {threshold}") # 뭔가 이상
  print("-"*60)
  # 예측값

  # 평가 데이터 정상
  cn7_predict_y = dropout_AE.predict(cn7_test_y)
  cn7_test_y_mse = np.mean(np.square(cn7_predict_y - cn7_test_y),axis=1)
  # 시각화
  plt.figure(3)
  plt.hist(cn7_test_y_mse,bins = 30)
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.title("testing normal data")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_y_anomalies = len(cn7_test_y_mse[cn7_test_y_mse > threshold])
  fn = cn7_test_y_anomalies
  # 정상을 정상으로 판단
  tp = len(cn7_test_y_mse)-fn
  print(f"불량/정상 갯수 : {fn}")
  print(f"정상/정상 갯수 : {tp}")
  
 # 평가 데이터 불량

  cn7_predict_n = dropout_AE.predict(cn7_test_n)
  cn7_test_n_mse = np.mean(np.square(cn7_predict_n - cn7_test_n),axis=1)
  # 시각화
  plt.figure(4)
  plt.hist(cn7_test_n_mse,bins = 30)
  plt.title("testing abnormal data")
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_n_anomalies = len(cn7_test_n_mse[cn7_test_n_mse > threshold])
  tn = cn7_test_n_anomalies
  # 실제 불량인데 정상으로 판단한 데이터
  fp = len(cn7_test_n_mse)-tn
  print(f"불량/불량 갯수 : {tn}")
  print(f"정상/불량 갯수 : {fp}")
  return (epoch,batch,fn,tp,tn,fp)


# In[ ]:





# In[ ]:


def allsummery(epochs,batches,it):
  arr = []
  for i in epochs:
    for j in batches:
      for k in range(it):
        print("-"*60)
        print(f"{k}th iteration epoch: {i}, batch: {j}")
        t = AE(i,j)
        arr.append(t)
  df = pd.DataFrame(arr,columns=["epoch","batch","fn","tp","tn","fp"])
  return df
relu_201020 = allsummery([100,50,30],[30,20,10],30)


# In[ ]:





# In[ ]:


relu_201020.head()


# In[ ]:


relu_201020["precision"] = relu_201020["tp"]/(relu_201020["tp"]+relu_201020["fp"])
relu_201020["recall"] = relu_201020["tp"]/(relu_201020["tp"]+relu_201020["fn"])
relu_201020["accuracy"] = (relu_201020["tp"]+relu_201020["tn"])/(relu_201020["tp"]+relu_201020["fp"]+relu_201020["fn"]+relu_201020["tn"])
relu_201020["F1"] = 2*relu_201020["recall"]*relu_201020["precision"]/(relu_201020["recall"]+relu_201020["precision"])


# In[ ]:


relu_201020.groupby(["epoch","batch"]).mean()


# In[ ]:


# 7:3 input 다르게
cn7_train = cn7_Y[:3000] # 아직 결과 데이터 안지운 상태 (둘다)
cn7_test_y = cn7_Y[3000:]
cn7_test_n =cn7_N

def AE(epoch, batch):
  # 인코더
  dropout_encoder = Sequential([
                                
      Dropout(0.3),
      Dense(20,activation="relu"), 
      Dense(10,activation="relu")])
  # 디코더
  dropout_decoder = Sequential([
      Dense(20,activation="relu",input_shape=[10]),
      Dense(cn7_train.shape[1],activation="sigmoid")])
  dropout_AE = Sequential([dropout_encoder,dropout_decoder])

  # 손실함수 옵티마이저 정의
  dropout_AE.compile(loss="mse",optimizer=Adam(lr=0.03),metrics=["accuracy"])
  # 모델훈련
  history = dropout_AE.fit(cn7_train,cn7_train,batch_size=batch,epochs=epoch,validation_split=0.2,callbacks=[EarlyStopping(monitor="val_loss",patience=10,mode="min")])
  plt.figure(1)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["accuracy"],label="Training Acc")
  plt.plot(history.history["val_accuracy"],label="Validation Acc")
  plt.legend()
  plt.figure(2)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["loss"],label="Training Loss")
  plt.plot(history.history["val_loss"],label="Validation Loss")
  plt.legend()
  plt.show()

  cn7_train_pred = dropout_AE.predict(cn7_train)
  cn7_train_loss = np.mean(np.square(cn7_train_pred - cn7_train),axis=1)
  threshold = np.mean(cn7_train_loss)+1.96*np.std(cn7_train_loss)
  print(f"복원 오류 임계치: {threshold}") # 뭔가 이상
  print("-"*60)
  # 예측값

  # 평가 데이터 정상
  cn7_predict_y = dropout_AE.predict(cn7_test_y)
  cn7_test_y_mse = np.mean(np.square(cn7_predict_y - cn7_test_y),axis=1)
  # 시각화
  plt.figure(3)
  plt.hist(cn7_test_y_mse,bins = 30)
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.title("testing normal data")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_y_anomalies = len(cn7_test_y_mse[cn7_test_y_mse > threshold])
  fn = cn7_test_y_anomalies
  # 정상을 정상으로 판단
  tp = len(cn7_test_y_mse)-fn
  print(f"불량/정상 갯수 : {fn}")
  print(f"정상/정상 갯수 : {tp}")
  
 # 평가 데이터 불량

  cn7_predict_n = dropout_AE.predict(cn7_test_n)
  cn7_test_n_mse = np.mean(np.square(cn7_predict_n - cn7_test_n),axis=1)
  # 시각화
  plt.figure(4)
  plt.hist(cn7_test_n_mse,bins = 30)
  plt.title("testing abnormal data")
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_n_anomalies = len(cn7_test_n_mse[cn7_test_n_mse > threshold])
  tn = cn7_test_n_anomalies
  # 실제 불량인데 정상으로 판단한 데이터
  fp = len(cn7_test_n_mse)-tn
  print(f"불량/불량 갯수 : {tn}")
  print(f"정상/불량 갯수 : {fp}")
  return (epoch,batch,fn,tp,tn,fp)
  #df_new = pd.DataFrame([(epoch,batch,fn,tp,tn,fp)])
  #relu_201020=relu_201020.append(df_new,ignore_index=True)

def allsummery(epochs,batches,it):
  arr = []
  for i in epochs:
    for j in batches:
      for k in range(it):
        print("-"*60)
        print(f"{k}th iteration epoch: {i}, batch: {j}")
        t = AE(i,j)
        arr.append(t)
  df = pd.DataFrame(arr,columns=["epoch","batch","fn","tp","tn","fp"])
  return df


# In[ ]:


relu_201020_2 = allsummery([100,50,30],[30,20,10],30)
relu_201020_2["precision"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fp"])
relu_201020_2["recall"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fn"])
relu_201020_2["accuracy"] = (relu_201020_2["tp"]+relu_201020_2["tn"])/(relu_201020_2["tp"]+relu_201020_2["fp"]+relu_201020_2["fn"]+relu_201020_2["tn"])
relu_201020_2["F1"] = 2*relu_201020_2["recall"]*relu_201020_2["precision"]/(relu_201020_2["recall"]+relu_201020_2["precision"])
relu_201020_2.head()


# In[ ]:


relu_201020_2.groupby(["epoch","batch"]).mean()


# In[ ]:


# no early stopping 
# 20(relu)-10(relu)-20(relu)-24(sigmoid-output)
cn7_train = cn7_Y[:2400] # 아직 결과 데이터 안지운 상태 (둘다)
cn7_test_y = cn7_Y[2400:]
cn7_test_n =cn7_N


def AE_aug(epoch, batch):
  # 인코더
  dropout_encoder = Sequential([
                                
      Dropout(0.3),
      Dense(20,activation="relu"), 
      Dense(10,activation="relu"),
      Dense(5,activation="relu")
      ])
  # 디코더
  dropout_decoder = Sequential([
      Dense(10,activation="relu",input_shape=[5]),
      Dense(20,activation="relu",input_shape=[10]),
      Dense(cn7_train.shape[1],activation="sigmoid")])
  dropout_AE = Sequential([dropout_encoder,dropout_decoder])

  # 손실함수 옵티마이저 정의
  dropout_AE.compile(loss="mse",optimizer=Adam(lr=0.03),metrics=["accuracy"])
  # 모델훈련
  history = dropout_AE.fit(cn7_train,cn7_train,batch_size=batch,epochs=epoch,validation_split=0.2,callbacks=[EarlyStopping(monitor="val_loss",patience=10,mode="min")])
  plt.figure(1)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["accuracy"],label="Training Acc")
  plt.plot(history.history["val_accuracy"],label="Validation Acc")
  plt.legend()
  plt.figure(2)
  plt.figure(figsize=(10,3))
  plt.plot(history.history["loss"],label="Training Loss")
  plt.plot(history.history["val_loss"],label="Validation Loss")
  plt.legend()
  plt.show()

  cn7_train_pred = dropout_AE.predict(cn7_train)
  cn7_train_loss = np.mean(np.square(cn7_train_pred - cn7_train),axis=1)
  threshold = np.mean(cn7_train_loss)+1.96*np.std(cn7_train_loss)
  print(f"복원 오류 임계치: {threshold}") # 뭔가 이상
  print("-"*60)
  # 예측값

  # 평가 데이터 정상
  cn7_predict_y = dropout_AE.predict(cn7_test_y)
  cn7_test_y_mse = np.mean(np.square(cn7_predict_y - cn7_test_y),axis=1)
  # 시각화
  plt.figure(3)
  plt.hist(cn7_test_y_mse,bins = 30)
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.title("testing normal data")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_y_anomalies = len(cn7_test_y_mse[cn7_test_y_mse > threshold])
  fn = cn7_test_y_anomalies
  # 정상을 정상으로 판단
  tp = len(cn7_test_y_mse)-fn
  print(f"불량/정상 갯수 : {fn}")
  print(f"정상/정상 갯수 : {tp}")
  
 # 평가 데이터 불량

  cn7_predict_n = dropout_AE.predict(cn7_test_n)
  cn7_test_n_mse = np.mean(np.square(cn7_predict_n - cn7_test_n),axis=1)
  # 시각화
  plt.figure(4)
  plt.hist(cn7_test_n_mse,bins = 30)
  plt.title("testing abnormal data")
  plt.xlabel("test mse loss")
  plt.ylabel("no of samples")
  plt.show()

  #불량으로 판단한 데이터 확인
  cn7_test_n_anomalies = len(cn7_test_n_mse[cn7_test_n_mse > threshold])
  tn = cn7_test_n_anomalies
  # 실제 불량인데 정상으로 판단한 데이터
  fp = len(cn7_test_n_mse)-tn
  print(f"불량/불량 갯수 : {tn}")
  print(f"정상/불량 갯수 : {fp}")
  return (epoch,batch,fn,tp,tn,fp)
  #df_new = pd.DataFrame([(epoch,batch,fn,tp,tn,fp)])
  #relu_201020=relu_201020.append(df_new,ignore_index=True)

  '''cn7_true = np.concatenate([ np.zeros(len(cn7_test_y_anomalies)), np.ones(len(cv7_test_n_anomalies))])
  cn7_prediction = np.concatenate([cv7_test_y_anomalies,cv7_test_n_anomalies])

  from sklearn.metrics import confusion_matrix
  print(confusion_matrix(cn7_true,cn7_prediction))'''

import time

def allsummery(epochs,batches,iteration):
  arr = []
  time_elapsed = []
  for i in epochs:
    for j in batches:
      for k in range(iteration):
        print("-"*60)
        print(f"{k}th batch: {j} epoches: {i}")
        start = time.time()
        t = AE_aug(i,j)
        end = time.time()
        time_elapsed.append(end-start)
        arr.append(t)
  df = pd.DataFrame(arr,columns=["epoch","batch","fn","tp","tn","fp"])
  df["elapsed time"] = time_elapsed
  return df
relu_201020_nostop = allsummery([100,50,30],[30,20,10],20)


# In[ ]:


relu_201020_nostop = relu_201020_2
relu_201020_2["precision"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fp"])
relu_201020_2["recall"] = relu_201020_2["tp"]/(relu_201020_2["tp"]+relu_201020_2["fn"])
relu_201020_2["accuracy"] = (relu_201020_2["tp"]+relu_201020_2["tn"])/(relu_201020_2["tp"]+relu_201020_2["fp"]+relu_201020_2["fn"]+relu_201020_2["tn"])
relu_201020_2["F1"] = 2*relu_201020_2["recall"]*relu_201020_2["precision"]/(relu_201020_2["recall"]+relu_201020_2["precision"])
relu_201020_2.head()


# In[ ]:


relu_201020_2.groupby(["epoch","batch"]).mean()

