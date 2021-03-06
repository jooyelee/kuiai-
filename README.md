# KUIAI 해커톤 1회

산업 현장에서 불량 찾기 --> abnormal detection

www.kamp-ai.kr

### 산업 데이터 
1. 라벨링 하는데 돈이 들어서 비지도 학습 해야됨(하지만 성능평가를 위해 라벨링 된 데이터 사용함)
2. 데이터 불균형: 불량 데이터 수집 확률 낮음
3. 컴퓨터 성능: 큰 모델 못돌림/ 학습시간 예측시간 짧아야됨
4. 검출 성능 문제: 검출 성능이 아주~ 좋아야지 실제 사용 가능 / 아니면 역효과 남

예시
1. isolation-forest
2. autoencoder
3. one-class SVM

Outlier detection vs Novelty detection
training set: 정상+비정상 vs 오직 정상

문제 데이터 : 사출 &  용해 탱크 데이터





### 과정

1. 데이터셋 자체를 변형해볼까(resampling)
https://joonable.tistory.com/27
- oversampling
SMOTE: synthtic minority over-sampling technique
MSNOTE: modified synthtic minority over-sampling technique

더... borderline smote / adasyn

2. 정상 데이터 만으로 학습시켜야됨-->(control chart phase1 생각) 
--> 관리도를 이용하면 안될까? 
--> 근데 관리도라는게 정상과 비정상 영역을 그냥 line으로 나누는거자나 그렇게 따지면 다른 novelty detection방법도 정상 영역을 미리 구분짓고 거기에 속하냐 안속하냐를 보는건데 관리도처럼 line으로 하는게 정확도가 제일 떨어질거자나
라고 생각해서 포기

3. 비지도 학습의 여러 모델을 활용해서 해보자(abnormal detection에 강한 3가지중 isolation-forest / autoencoder) 
 

<사출데이터> - 나 해주언니
<융해데이터> - 해인 미섭언니

< 사출데이터 >
원리-불량요인 보면 많은 요인들이 복합적으로 작용 : 나는 최대한 많은 feature를 활용하자고 생각 — 그래서 auto encoder사용

기계의 부품별로 해봄 :한 부품 굿/ 딴 부품 뱃

autoencoder : 압축 – 복원: 복원시 차이 이용해 판별
feature수 23:   20-10-20 relu 사용해봄
