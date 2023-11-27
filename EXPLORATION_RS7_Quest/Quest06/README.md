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
    
        > 문제에서 요구한 배경 블러, 배경 합성, 동물 사진 합성 등의 요구사항을 모두 충족하였다
        
    
        ![image](https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/57254153-e778-42dc-84b7-18b04c08b1de)
    
        ![image](https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/868dce95-e5c7-4163-857d-589cf81cae33)
    
        ![image](https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/421e17ae-1a3b-4ad4-beef-91183ba2e355)
    
        
        <br/>

        > 문제에서 요구한 문제점 지적 및 솔루션 제안을 수행하였습니다.

        ![image](https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/210eb342-3576-4604-aa39-aa5f9e0c500e)

      <br/>
      <br/>

    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
        <br/>
        
        > 각 코드를 작성한 이유와 역할에 대하여 주석을 통하여 상세하게 설명하였습니다
         
        ```
        # cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
        # cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니
        # 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경) 
        img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        
        # cv2.bitwise_not(): 이미지가 반전됩니다. 배경이 0 사람이 255 였으나
        # 연산을 하고 나면 배경은 255 사람은 0입니다.
        img_bg_mask = cv2.bitwise_not(img_mask_color)
        
        # cv2.bitwise_and()을 사용하면 배경만 있는 영상을 얻을 수 있습니다.
        # 0과 어떤 수를 bitwise_and 연산을 해도 0이 되기 때문에 
        # 사람이 0인 경우에는 사람이 있던 모든 픽셀이 0이 됩니다. 결국 사람이 사라지고 배경만 남아요!
        img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
        plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
        plt.show()
        ```
    
        <br/>

        

    
        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
        <br/>
        
        > 이미지 합성을 수행하기 위하여 세그멘테이션 및 컬러 마스크 생성 과정을 상세하게 기록하였습니다
        
        ![image](https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/0d50a6fe-414a-444f-92b4-6faef6a8d9e7)
     
        ![image](https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/1b385b3f-dfc7-4b5f-943f-cab00ea519ef)


        <br/>
        
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
        <br/>
     
        > 회고록을 통해 과제 수행 시 어려웠던 점과 나아가야 할 점에 대하여 언급하였습니다
     
        ```
        포토샾 합성도 해본 적이 없는데 코드로 간단하게 사진을 아웃포커싱 처리를 하고 크로마키 합성도 하면서 문득 기술의 발전에 대해 대단하다고 느끼게 되는 경험이였다.  

        semantic segmentation mask의 오류를 보완할 수 있는 좋은 솔루션을 이유와 함께 제시를 하는 부분에서 원인이 무엇인지는 이해를 하였으나 depth를 추가하는 부분에 대한 이해가 부족하다고 느꼈는데 아직 공부할 것이 매우 많다는 사실을 다시 깨닫게 되었다.  

        마지막으로 이러한 문제를 해결하기 위한 방법을 찾아보면서 굉장히 많은 해결방법이 있다는 것을 보며 내가 앞으로 aiffel 프로젝트를 진행하고 나아가 현업에서 일하게 되더라도 여러가지 해결책을 찾아보는 것이 중요하겠다는 생각을 하게 되었다.
        ```

        <br/>
        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
     
      <br/>
      
      > 변수의 역할과 목적을 명확하게 알 수 있는 변수명을 사용하였습니다
      
      ```
      model_dir = os.getenv('HOME')+'/aiffel/human_segmentation/models'
      model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
      # PixelLib가 제공하는 모델의 url입니다
      model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'
      # 다운로드를 시작합니다
      urllib.request.urlretrieve(model_url, model_file) # urllib 패키지 내에 있는 request 모듈의 urlr
      ```

      <br/>


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

**실력도 좋으신데 성품도 따뜻하신 승제님 🐻💕**   
문제 해결을 위해 실험을 거듭하고 고민하신 흔적이 잘 남아있어서 리뷰어로서 코드 보기 좋았습니다   
실험 기록을 잘 남겨놓는 모습을 저도 본받고 싶습니다 😙😙
