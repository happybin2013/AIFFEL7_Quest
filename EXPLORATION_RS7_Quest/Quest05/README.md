# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이승제
- 리뷰어 : 강다은


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
     
        <br/>
     
        > 최종적으로 test accuracy 85% 이상 달성
        
        ![image](https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/5c3bc7e3-28cb-4d18-be7e-4397cfd7268d)

        <br/>

        
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
        <br/>
     
        >코드를 실행하는 과정과 그렇게 설계한 이유를 주석을 통해 순차적으로 잘 설명하였습니다.  
        
        ```
        total_data_text = list(X) + list(X)

        # 텍스트데이터 문장길이의 리스트를 생성한 후
        num_tokens = [len(tokens) for tokens in total_data_text]
        num_tokens = np.array(num_tokens)
        
        # 문장길이의 평균값, 최대값, 표준편차를 계산해 본다. 
        print('문장길이 평균 : ', np.mean(num_tokens))
        print('문장길이 최대 : ', np.max(num_tokens))
        print('문장길이 표준편차 : ', np.std(num_tokens))
        
        # 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,  
        max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
        maxlen = int(max_tokens)
        print('pad_sequences maxlen : ', maxlen)
        print(f'전체 문장의 {np.sum(num_tokens < max_tokens) / len(num_tokens)}%가 maxlen 설정값 이내에 포함
        ```
        <br/>

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
        <br/>
     
        >다양한 인자값을 시도하면서 목표 달성에 실패한 기록들을 보기 쉽게 정리해두었습니다.

        <img width="653" alt="image" src="https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/1a4c8619-41bc-49f5-87ce-84f3ee9e1d88">

        <br/>
        <br/>

        
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
        <br/>
     
        >프로젝트를 수행하면서 어려웠던 점, 해결 방식, 느낀 점 등을 간결하게 작성하였습니다.
        ```
        회고
        비정형 데이터를 가지고 뭔가의 유사도를 측정한다는 것이 정말 신기했던 프로젝트이다. 프로젝트는 어렵고 힘들었지만 그래도 여러 모델들을 사용하고 옵션 값을 바꿔보면서 어떤 영향을 주는지에 대해서 좀 더 자세하게 공부할 수 있는 시간이 되었던 것 같습니다.
        ```
        <br/>
        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

      <br/>
     
        >코드가 간결하고, 변수명/함수명 등을 간결하게 작성하였습니다.
        >특히 parameter가 다수 포함된 함수 호출 시, 줄바꿈을 통해 가독성을 확보하였습니다.
        ```
        #vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
        #word_vector_dim = 100  # 워드 벡터의 차원 수 
        
        # 모델 구성
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
        model.add(tf.keras.layers.LSTM(64))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경 가능)
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.
        
        model.summary()


        # 학습
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
                      
        epochs=3  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 
        
        historys = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=64,
                            validation_data=(X_val, y_val),
                            verbose=2)
        ```
        <br/>
        <br/>


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

**성실하고 상냥하신 승제님** 💕  
실험과 고민 과정이 착실하게 담긴 좋은 코드를 리뷰할 기회를 주셔서 감사합니다.  
개발과정의 기록을 남기는 방법에 의문이 많았는데 덕분에 많이 배웠습니다.  
코드를 통해 개발과정을 지켜보다보니 오랜 고민 끝에 멋진 성능을 발휘하신 결말에 저까지 뿌듯했습니다 😄ㅎㅎ
