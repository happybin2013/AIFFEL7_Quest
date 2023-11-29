# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이승제
- 리뷰어 : 윤빛나

루브릭

아래의 기준을 바탕으로 프로젝트를 평가합니다.

pix2pix 모델 학습을 위해 필요한 데이터셋을 적절히 구축하였다. 데이터 분석 과정 및 한 가지 이상의 augmentation을 포함한 데이터셋 구축 과정이 체계적으로 제시되었다.

pix2pix 모델을 구현하여 성공적으로 학습 과정을 진행하였다. U-Net generator, discriminator 모델 구현이 완료되어 train_step의 output을 확인하고 개선하였다.

학습 과정 및 테스트에 대한 시각화 결과를 제출하였다. 10 epoch 이상의 학습을 진행한 후 최종 테스트 결과에서 진행한 epoch 수에 걸맞은 정도의 품질을 확인하였다.



# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**

1. 한 가지 이상의 augmentation을 사용하였습니다.

<img width="664" alt="스크린샷 2023-11-29 12 34 03" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/4edbd051-b324-44a0-9f60-8d44cc65c839">

2. U-Net generator, discriminator 모델 구현이 완료되었습니다.

<img width="713" alt="스크린샷 2023-11-29 12 35 12" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/400ca576-acde-4bdd-bfdb-cbf48b87bf95">

<img width="868" alt="스크린샷 2023-11-29 12 35 46" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/779f6928-dc6e-4f24-b011-d0be8369ee8a">

3. 10 epoch 이상의 학습을 진행한 후 테스트에 대한 시각화 결과를 제출하셨습니다.

<img width="698" alt="스크린샷 2023-11-29 12 44 10" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/65ac64c7-1c76-4024-a86a-fe92dc4b2a47">

<img width="268" alt="스크린샷 2023-11-29 12 43 03" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/c0f1c1cb-1a85-4619-a326-ecb27ff2258e">


    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**

코드 안에 적절한 주석처리로 복잡한 코드도 알아보기 쉽도록 잘 작성되었습니다.

<img width="856" alt="스크린샷 2023-11-29 12 47 43" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/dff3abac-f650-43ab-bc2d-f2ff3bb28384">

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**

에폭을 여러차례 돌려보며 결과를 비교하고 그 결과물을 상세하게 남겨주셨습니다.

<img width="903" alt="스크린샷 2023-11-29 12 50 44" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/e88a0cc1-5283-41d2-8c99-9ee19d6c8b85">

<img width="886" alt="스크린샷 2023-11-29 12 49 43" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/c4554912-02a0-47d1-b005-89552ab0e1b8">

<img width="898" alt="스크린샷 2023-11-29 12 49 48" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/99f4cb75-5220-4e57-a783-c71c223278d2">

<img width="894" alt="스크린샷 2023-11-29 12 49 53" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/8c6cea74-2ff5-40bf-a485-8eb0d92bc0d8">

<img width="891" alt="스크린샷 2023-11-29 12 49 57" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/9f11a76e-fde3-4e92-a821-1275d4e55181">

<img width="886" alt="스크린샷 2023-11-29 12 50 01" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/352da084-9411-4665-976f-70f7e24b9362">

<img width="883" alt="스크린샷 2023-11-29 12 50 07" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/bbe2c853-1507-4d46-8ca2-da395c72ebde">


- [x]  **4. 회고를 잘 작성했나요?**

이번 프로젝트를 하면서 느꼈던 점과 알게 된 점 그리고 아쉬웠던 점들이 적절하게 잘 작성되어 있습니다.

<img width="890" alt="스크린샷 2023-11-29 12 52 58" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/97ec8cfb-7324-46c8-9668-ff548dbac798">

        
- [x]  **5. 코드가 간결하고 효율적인가요?**

for문을 활용하여 DiskBlock을 잘 쌓고 코드가 간결하고 시각적으로 보기 좋게 잘 작성되어 있습니다.

<img width="883" alt="스크린샷 2023-11-29 12 56 50" src="https://github.com/happybin2013/AIFFEL7_Quest/assets/100587126/bf2d066f-c17a-4b02-9a82-7720d71ba8da">


프로젝트 하시느라 고생 많으셨습니다! 항상 화이팅하세요!!

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
