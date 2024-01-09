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

       > segmentation을 위한 U-net과 U-net++ 모델을 구현하고, 실험 수행을 통해 두 모델의 성능을 비교하였다.

       <br/>

       <img width="673" alt="image" src="https://github.com/DiANA-KANG/AIFFEL7_Quest_LSJ/assets/149550222/f6f0d22d-df2e-4440-9870-ac0752ee8f00">

       <br/>
       <br/>
       <br/>



- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

       <br/>

       > doc string과 주석을 통해 함수의 기능과 작동순서를 간결하게 명시하였다.

       <br/>

       ```
       class KittiGenerator(tf.keras.utils.Sequence):
           '''
           KittiGenerator는 tf.keras.utils.Sequence를 상속받습니다.
           우리가 KittiDataset을 원하는 방식으로 preprocess하기 위해서 Sequnce를 커스텀해 사용합니다.
           '''
           def __init__(self,
                       dir_path,
                       batch_size=16,
                       img_size=(224, 224, 3),
                       output_size=(224, 224),
                       is_train=True,
                       augmentation=None):
           '''
           dir_path: dataset의 directory path입니다.
           batch_size: batch_size입니다.
           img_size: preprocess에 사용할 입력이미지의 크기입니다.
           output_size: ground_truth를 만들어주기 위한 크기입니다.
           is_train: 이 Generator가 학습용인지 테스트용인지 구분합니다.
           augmentation: 적용하길 원하는 augmentation 함수를 인자로 받습니다.
           '''
           self.dir_path = dir_path
           self.batch_size = batch_size
           self.is_train = is_train
           self.augmentation = augmentation
           self.img_size = img_size
           self.output_size = output_size

           # load_dataset()을 통해서 kitti dataset의 directory path에서 라벨과 이미지를 확인합니다.
           self.data = self.load_dataset()
       ```

       <br/>


       
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

       <br/>

       > backbone 구조를 시도해보거나, epochs 횟수에 따라 성능을 확인하는 등 부가적인 실험을 수행하였다.

       <br/>

       ```
       backbone을 해보려다가 input size문제와 많은 양의 코드로 시간이 부족해 실패했다.
       https://github.com/MrGiovanni/UNetPlusPlus <- unet++ 깃헙
       https://www.kaggle.com/code/meaninglesslives/nested-unet-with-efficientnet-encoder/notebook#Useful-Model-Blocks <- backbone kaggle
       ```
       ```
       - 1번은 epoch을 40번 진행한 사진이고 2번은 epoch 100 번 진행한 사진인데 다른 경우는 거의 비슷하거나 unet이 iou 0.01정도 잘 나오는 것을 볼 수 있었다.
       ```

       <br/>


       
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

       <br/>

       > 회고록을 통해 프로젝트를 수행하면서 배운 점과 아쉬운 점, 앞으로 보완할 점에 대하여 언급하였다.

       <br/>

       ```
       오늘 프로젝트를 진행하면서 새로운 customl_loss를 사용해 봤는데 흔히 쓰는 loss 처럼 일정하게 내려가는 느낌은 아니였던 것 같다. 이런 특이한 것들을 좀 찾아보고 알아보면 좋을 것 같다.
       그리고 아쉬운 점은 backbone을 사용한 모델을 추가로 실험해 보면 좋을 것 같고 ealrystop을 넣어서 학습이 좀 빨리 끝난 것 같아서 풀고 좀 더 학습을 시키면 좋을 것 같다.
       그리고 문제점이 있었는데 loss를 변경해서 진행했더니 save가 안되는 문제가 있었다. 이 문제도 해결해 봐야겠다.
       ```

       <br/>


       
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

       <br/>

       > 모델 구조 내에서 반복되는 부분을 block 함수로서 별도 구현하여 코드 중복을 최소화하였다.

       <br/>
       
       ```
       # Convolutional block
       def conv_block_plus(input_tensor, num_filters, dropout_rate=0.0):
           x = layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
           x = layers.Activation("relu")(x)

           if dropout_rate > 0:
               x = layers.Dropout(dropout_rate)(x)

           x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
           x = layers.Activation("relu")(x)
           return x

       # Encoder block
       def encoder_block_plus(input_tensor, num_filters):
           x = conv_block_plus(input_tensor, num_filters)
           p = layers.MaxPooling2D((2, 2))(x)
           return x, p

       # Decoder block using Transpose2D
       def decoder_block_plus(input_tensor, concat_tensor, num_filters):
           x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
           x = layers.concatenate([x, concat_tensor], axis=-1)
           x = conv_block_plus(x, num_filters)
           return x

       # Build U-Net++ model
       def build_unetplusplus(input_shape=(224, 224, 3), num_classes=1):
           inputs = tf.keras.Input(shape=input_shape)

           # Encoder
           x1, p1 = encoder_block_plus(inputs, 64)
           x2, p2 = encoder_block_plus(p1, 128)
           x3, p3 = encoder_block_plus(p2, 256)
           x4, p4 = encoder_block_plus(p3, 512)

           # Bridge
           b = conv_block_plus(p4, 1024, dropout_rate=0.5)

           # Decoder
           d1 = decoder_block_plus(b, x4, 512)
           d2 = decoder_block_plus(d1, x3, 256)
           d3 = decoder_block_plus(d2, x2, 128)
           d4 = decoder_block_plus(d3, x1, 64)

           # Nested connections
           d1_nest = decoder_block_plus(x4, x3, 256)
           d2_nest = decoder_block_plus(d1_nest, x2, 128)
           d3_nest = decoder_block_plus(d2_nest, x1, 64)

           d2_nest2 = decoder_block_plus(x3, x2, 128)
           d3_nest2 = decoder_block_plus(d2_nest2, x1, 64)

           d3_nest3 = decoder_block_plus(x2, x1, 64)

           # Final layer
           final_concat = layers.concatenate([d4, d3_nest, d3_nest2, d3_nest3], axis=-1)
           outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(final_concat)

           model = models.Model(inputs, outputs, name="U-NetPlusPlus")
           return model
       ```
       
       <br/>
       <br/>
       <br/>

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
