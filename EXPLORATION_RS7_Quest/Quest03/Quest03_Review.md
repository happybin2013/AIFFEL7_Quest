# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이승제
- 리뷰어 : 박준


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부

![kaggle](https://github.com/happybin2013/AIFFEL7_Quest/assets/7679722/cb949bf4-86d9-4415-b586-29673697857b)

캐글에 제출해 주신 결과를 첨부해 주셨습니다.
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인

각 과정별로 어떤 것들이 진행 되느니 잘 알 수 있었다.
```
print("#8 데이터 분포, 상관관계 확인")

def correlation_plot():
count = 0
columns = train.columns
fig, ax = plt.subplots(10, 2, figsize=(12, 30))   # 가로스크롤 때문에 그래프 확인이 불편하다면 figsize의 x값을 조절해 보세요.

train['price'] = np.log1p(train['price'])

for row in range(10):
    for col in range(2):
        try:
            # sns.regplot(np.log1p(train[columns[count]]), train['price'], ax=ax[row][col])
            sns.regplot(train[columns[count]], train['price'], ax=ax[row][col])
            ax[row][col].set_title(columns[count], fontsize=15)
            plt.tight_layout()
        except:
            print(columns[count])
        count += 1
        if count == 20:
            break
````

    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

아래 코드와 같이 gridsearch 함수에대해 간단한 주석이 있었습니다.
```
def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    # GridSearchCV 모델로 초기화
    grid_model = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', \
                          cv=5, verbose=verbose, n_jobs=n_jobs)
    
    # 모델 fitting
    grid_model.fit(train2, y)
    
    # 결과값 저장
    params = grid_model.cv_results_['params']
    score = grid_model.cv_results_['mean_test_score']
    
    # 데이터 프레임 생성
    results = pd.DataFrame(params)
    results['score'] = score
    
    # RMSLE 값 계산 후 정렬
    results['RMSLE'] = np.sqrt(-1 * results['score'])
    results = results.sort_values('RMSLE')
    
    return results
```
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

새로운 그래디언트 부스팅 모델을 활용해 보신점이 멋지십니다.
```
!pip install catboost
```
![catboost.png](https://github.com/happybin2013/AIFFEL7_Quest/assets/7679722/d6c7c28a-51b5-415c-82ea-059995c944fe)

- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

![회고](https://github.com/happybin2013/AIFFEL7_Quest/assets/7679722/1df71c23-d0a2-42e5-9373-7f2418c8662c)

교훈과 아쉬운 점, 앞으로의 목표에 대해서 잘 적어주셨다.
        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

![code](https://github.com/happybin2013/AIFFEL7_Quest/assets/7679722/22d61387-ee35-452e-9e67-aa4c081b288c)

아주 간결하고 깔끔하게 전체 훈련 진행과정을 볼수 있는 코드였습니다.


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
앙상블 기법과 다양한 하이퍼파라미터 활용에 대해 말씀드렸습니다.
특히 `num_boost_round`에 대해 수치를 높여보면서 실험해보시면 좋을것 같다고 전달했습니다.
https://gorakgarak.tistory.com/1285

