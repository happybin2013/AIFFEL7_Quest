#!/usr/bin/env python
# coding: utf-8

# # Quest08: Segmentation map으로 도로 이미지 만들기
# 
# ### 평가문항에 맞춰서 작성
# **1. pix2pix 모델 학습을 위해 필요한 데이터셋을 적절히 구축하였다.**  
# -> 데이터 분석 과정 및 한 가지 이상의 augmentation을 포함한 데이터셋 구축 과정이 체계적으로 제시되었다.
# 
# **2. pix2pix 모델을 구현하여 성공적으로 학습 과정을 진행하였다.**  
# -> U-Net generator, discriminator 모델 구현이 완료되어 train_step의 output을 확인하고 개선하였다.
# 
# **3. 학습 과정 및 테스트에 대한 시각화 결과를 제출하였다.**  
# -> 10 epoch 이상의 학습을 진행한 후 최종 테스트 결과에서 진행한 epoch 수에 걸맞은 정도의 품질을 확인하였다.

# # Import

# In[1]:


import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import image
from tensorflow.keras.preprocessing.image import random_rotation
from tensorflow import data
from tensorflow.keras import layers, Input, Model
from tensorflow.keras import losses
from tensorflow.keras import optimizers


# # gpu 환경설정

# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 사용 제한을 위한 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # 특정 GPU만 사용하도록 설정
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # 프로그램 시작 후에는 GPU 설정을 변경할 수 없으므로
        # 런타임 오류 발생 시 예외 처리가 필요함
        print(e)


# In[3]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# ---
# 
# # <span style="background-color:#E6E6FA">  1. 도로 이미지 데이터셋 가져오기</span>

# In[4]:


data_path = os.getenv('HOME')+'/aiffel/segmentation_map/data/cityscapes/train/'
print("number of train examples :", len(os.listdir(data_path)))


# In[5]:


plt.figure(figsize=(20,15))
for i in range(1, 7):
    f = data_path + os.listdir(data_path)[np.random.randint(800)]
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    plt.subplot(3,2,i)
    plt.imshow(img)


# ---
# 
# # <span style="background-color:#E6E6FA">  2. 데이터 확인 및 전처리</span>

# In[6]:


f = data_path + os.listdir(data_path)[0]
img = cv2.imread(f, cv2.IMREAD_COLOR)
print(img.shape)


# In[7]:


def normalize(x):
    x = tf.cast(x, tf.float32)
    return (x/127.5) - 1

def denormalize(x):
    x = (x+1)*127.5
    x = x.numpy()
    return x.astype(np.uint8)

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, 3)
    
    w = tf.shape(img)[1] // 2
    sketch = img[:, :w, :] 
    sketch = tf.cast(sketch, tf.float32)
    colored = img[:, w:, :] 
    colored = tf.cast(colored, tf.float32)
    return normalize(sketch), normalize(colored)

f = data_path + os.listdir(data_path)[1]
colored, sketch = load_img(f)

plt.figure(figsize=(10,7))
plt.subplot(1,2,1); plt.imshow(denormalize(sketch))
plt.subplot(1,2,2); plt.imshow(denormalize(colored))


# ---
# 
# # <span style="background-color:#E6E6FA">  3. 데이터에 한 가지 이상의 augmentation 방법을 적용  </span>

# In[8]:


@tf.function() # 빠른 텐서플로 연산을 위해 @tf.function()을 사용합니다. 
def apply_augmentation(sketch, colored):
    stacked = tf.concat([sketch, colored], axis=-1)
    
    _pad = tf.constant([[30,30],[30,30],[0,0]])
    if tf.random.uniform(()) < .5:
        padded = tf.pad(stacked, _pad, "REFLECT")
    else:
        padded = tf.pad(stacked, _pad, "CONSTANT", constant_values=1.)

    out = image.random_crop(padded, size=[256, 256, 6])
    
    out = image.random_flip_left_right(out)
    out = image.random_flip_up_down(out)
    
    if tf.random.uniform(()) < .5:
        degree = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        out = image.rot90(out, k=degree)
    
    return out[...,:3], out[...,3:]


# In[9]:


plt.figure(figsize=(15,13))
img_n = 1
for i in range(1, 13, 2):
    augmented_sketch, augmented_colored = apply_augmentation(sketch, colored)
    
    plt.subplot(3,4,i)
    plt.imshow(denormalize(augmented_sketch)); plt.title(f"Image {img_n}")
    plt.subplot(3,4,i+1); 
    plt.imshow(denormalize(augmented_colored)); plt.title(f"Image {img_n}")
    img_n += 1


# In[10]:


from tensorflow import data

def get_train(img_path):
    colored, sketch = load_img(img_path)
    colored, sketch = apply_augmentation(sketch, colored)
    return colored, sketch

train_images = data.Dataset.list_files(data_path + "*.jpg")
train_images = train_images.map(get_train).shuffle(100).batch(4)

sample = train_images.take(1)
sample = list(sample.as_numpy_iterator())
colored, sketch = (sample[0][0]+1)*127.5, (sample[0][1]+1)*127.5

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(sketch[0].astype(np.uint8))
plt.subplot(1,2,2); plt.imshow(colored[0].astype(np.uint8))


# ---
# 
# # <span style="background-color:#E6E6FA">  4.  U-Net Generator를 사용 </span>

# In[11]:


class EncodeBlock(layers.Layer):
    def __init__(self, n_filters, use_bn=True):
        super(EncodeBlock, self).__init__()
        self.use_bn = use_bn       
        self.conv = layers.Conv2D(n_filters, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(0.2)

    def call(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.batchnorm(x)
        return self.lrelu(x)

    
class DecodeBlock(layers.Layer):
    def __init__(self, f, dropout=True):
        super(DecodeBlock, self).__init__()
        self.dropout = dropout
        self.Transconv = layers.Conv2DTranspose(f, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
    def call(self, x):
        x = self.Transconv(x)
        x = self.batchnorm(x)
        if self.dropout:
            x = layers.Dropout(.5)(x)
        return self.relu(x)


# In[12]:


class UNetGenerator(Model):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        encode_filters = [64,128,256,512,512,512,512,512]
        decode_filters = [512,512,512,512,256,128,64]
        
        self.encode_blocks = []
        for i, f in enumerate(encode_filters):
            if i == 0:
                self.encode_blocks.append(EncodeBlock(f, use_bn=False))
            else:
                self.encode_blocks.append(EncodeBlock(f))
        
        self.decode_blocks = []
        for i, f in enumerate(decode_filters):
            if i < 3:
                self.decode_blocks.append(DecodeBlock(f))
            else:
                self.decode_blocks.append(DecodeBlock(f, dropout=False))
        
        self.last_conv = layers.Conv2DTranspose(3, 4, 2, "same", use_bias=False)
    
    def call(self, x):
        features = []
        for block in self.encode_blocks:
            x = block(x)
            features.append(x)
        
        features = features[:-1]
                    
        for block, feat in zip(self.decode_blocks, features[::-1]):
            x = block(x)
            x = layers.Concatenate()([x, feat])
        
        x = self.last_conv(x)
        return x
                
    def get_summary(self, input_shape=(256,256,3)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()


# In[13]:


UNetGenerator().get_summary()


# ---
# 
# # <span style="background-color:#E6E6FA">  5. Discriminator 구현하기 </span>

# In[14]:


class DiscBlock(layers.Layer):
    def __init__(self, n_filters, stride=2, custom_pad=False, use_bn=True, act=True):
        super(DiscBlock, self).__init__()
        self.custom_pad = custom_pad
        self.use_bn = use_bn
        self.act = act
        
        # Custom padding이 적용되면 ZeroPadding2D 레이어와 valid 컨볼루션을 사용, 아니면 same 컨볼루션 사용
        if custom_pad:
            self.padding = layers.ZeroPadding2D()
            self.conv = layers.Conv2D(n_filters, 4, stride, "valid", use_bias=False)
        else:
            self.conv = layers.Conv2D(n_filters, 4, stride, "same", use_bias=False)
        
        # Batch Normalization 레이어 사용 여부
        self.batchnorm = layers.BatchNormalization() if use_bn else None
        # LeakyReLU 사용 여부
        self.lrelu = layers.LeakyReLU(0.2) if act else None
        
    def call(self, x):
        # Custom padding이 적용되면 패딩 후 컨볼루션, 아니면 바로 컨볼루션
        if self.custom_pad:
            x = self.padding(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
                
        # Batch Normalization 레이어 사용 여부에 따라 적용
        if self.use_bn:
            x = self.batchnorm(x)
            
        # LeakyReLU 사용 여부에 따라 적용
        if self.act:
            x = self.lrelu(x)
        return x


# In[15]:


inputs = Input((128,128,32))
out = layers.ZeroPadding2D()(inputs)
out = layers.Conv2D(64, 4, 1, "valid", use_bias=False)(out)
out = layers.BatchNormalization()(out)
out = layers.LeakyReLU(0.2)(out)

Model(inputs, out).summary()


# In[16]:


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        filters = [64,128,256,512,1]
        self.blocks = [layers.Concatenate()] # 입력을 연결하는 Concatenate 레이어 추가
        
        for i, f in enumerate(filters):
            self.blocks.append(DiscBlock(
                n_filters=f, 
                stride=2 if i < 3 else 1, 
                custom_pad= i >= 3,
                use_bn= i != 0 and i != 4, 
                act= i < 4
            ))
        self.sigmoid = layers.Activation(tf.keras.activations.sigmoid)
    
    def call(self, x, y):
        # 입력된 x와 y를 Concatenate 레이어를 통해 연결하고, 각 DiscBlock을 통과
        out = self.blocks[0]([x, y])
        for block in self.blocks[1:]:
            out = block(out)
        return self.sigmoid(out) # 최종 결과에 sigmoid를 적용하여 확률값으로 변환
    #def call(self, x, y):
    #    out = self.block1([x, y])
    #    out = self.block2(out)
    #    out = self.block3(out)
    #    out = self.block4(out)
    #    out = self.block5(out)
    #    out = self.block6(out)
    #    return self.sigmoid(out)
    
    def get_summary(self, x_shape=(256,256,3), y_shape=(256,256,3)):
        x, y = Input(x_shape), Input(y_shape) 
        return Model((x, y), self.call(x, y)).summary() # 입력 형태를 고려하여 모델의 요약 정보 반환


# In[17]:


Discriminator().get_summary()


# In[18]:


x = tf.random.normal([1,256,256,3])
y = tf.random.uniform([1,256,256,3])

disc_out = Discriminator()(x, y)
plt.imshow(disc_out[0, ... ,0])
plt.colorbar()


# ---
# 
# # <span style="background-color:#E6E6FA">  6. 학습된 Generator를 이용해 테스트 </span>

# In[19]:


bce = losses.BinaryCrossentropy(from_logits=False)
mae = losses.MeanAbsoluteError()

def get_gene_loss(fake_output, real_output, fake_disc):
    l1_loss = mae(real_output, fake_output)
    gene_loss = bce(tf.ones_like(fake_disc), fake_disc)
    total_gen_loss = gene_loss + (100 * l1_loss) # <<< total 값 추가 이 값으로 gradient를 계산해야 학습이 정상적으로 진행됨.!!!!
    return total_gen_loss, gene_loss, l1_loss

def get_disc_loss(fake_disc, real_disc):
    return bce(tf.zeros_like(fake_disc), fake_disc) + bce(tf.ones_like(real_disc), real_disc)


# In[20]:


gene_opt = optimizers.Adam(2e-4, beta_1=.5, beta_2=.999)
disc_opt = optimizers.Adam(2e-4, beta_1=.5, beta_2=.999)


# In[21]:


# 가중치를 1회 업데이트

generator = UNetGenerator()
discriminator = Discriminator()

@tf.function
def train_step(sketch, real_colored):
    with tf.GradientTape() as gene_tape, tf.GradientTape() as disc_tape:
    # 이전에 배웠던 내용을 토대로 train_step을 구성해주세요.
        # Generator에 Sketch를 입력하여 생성된 이미지를 얻습니다.
        generated_colored = generator(sketch, training=True)

        # Discriminator에 실제 컬러 이미지와 생성된 컬러 이미지를 입력합니다.
        real_disc = discriminator(real_colored, sketch, training=True)
        fake_disc = discriminator(generated_colored, sketch, training=True)

        # Generator의 손실, Loss를 계산합니다.
        total_gen_loss, gene_loss, l1_loss = get_gene_loss(generated_colored, real_colored, fake_disc)

        # Discriminator의 손실을 계산합니다.
        disc_loss = get_disc_loss(fake_disc, real_disc)

    # Generator와 Discriminator의 Gradient를 계산합니다.
    gene_gradient = gene_tape.gradient(total_gen_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gene_opt.apply_gradients(zip(gene_gradient, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    return total_gen_loss, l1_loss, disc_loss


# In[ ]:


import time

EPOCHS = 100

step_check = []
loss_check = []

for epoch in range(1, EPOCHS+1):
    strt_time = time.time()
    for i, (sketch, colored) in enumerate(train_images):
        total_gen_loss, l1_loss, d_loss = train_step(sketch, colored)
                
        # 200회 반복마다 손실을 출력합니다.
        if (i+1) % 250 == 0:
            print(f"EPOCH[{epoch}] - STEP[{i+1}]                     \nTotal_gen_loss:{total_gen_loss.numpy():.4f}                     \nL1_loss:{l1_loss.numpy():.4f}                     \nDiscriminator_loss:{d_loss.numpy():.4f}")
            
        if (i+1) % 10 == 0: 
            step_check.append(epoch*(i+1))
            loss_check.append(round(total_gen_loss.numpy(),2))
            
    end_time = time.time()
    print(f"EPOCH[{epoch}] 소요시간: {end_time - strt_time:.5f} sec", end="\n\n")


# ### <span style="background-color:#fff5b1"> step당 loss 그래프 출력 </span> 

# In[ ]:


plt.plot(loss_check)
plt.title("Loss By Step")
plt.xlabel('Step/10')
plt.ylabel('Loss')
plt.show()


# ---
# 
# # <span style="background-color:#E6E6FA">  7. 스케치, 생성된 사진, 실제 사진 순서로 나란히 시각화 </span>

# In[ ]:


test_ind = 1

f = data_path + os.listdir(data_path)[test_ind] # "val" 폴더
colored, sketch = load_img(f)

pred = generator(tf.expand_dims(sketch, 0))
pred = denormalize(pred)

plt.figure(figsize=(20,10))
plt.subplot(1,3,1); plt.imshow(denormalize(sketch))
plt.subplot(1,3,2); plt.imshow(pred[0])
plt.subplot(1,3,3); plt.imshow(denormalize(colored))


# ---
# 
# # <span style="background-color:#C0FFFF"> 결과 </span>
# 
# 
# ![image-10.png](attachment:image-10.png) <br/> *프로젝트에서 진행한 loss 그래프* |![image.png](attachment:image.png)  <br/> *텐서 pixtopix https://www.tensorflow.org/tutorials/generative/pix2pix?hl=ko *
# --- | --- |
# 
# 위 사진처럼 loss 그래프가 나왔는데 왼쪽은 프로젝트로 진행한 loss이고 오른쪽은 텐서에서 학습을 진행한 loss그래프이다.  
# loss 그래프가 굉장히 왔다갔다 하는 것을 볼 수 있는데 처음에는 학습이 잘 안되는 것 같았지만 step이 진행되면서 평균적으로 점점 낮아지는 것을 확인 할 수 있었다.  
# 
# <br/>
# <br/>
# <br/>
# 
# ![image-13.png](attachment:image-13.png)
# 
# 위 사진은 epoch을 100으로 학습한 모델을 가지고 출력한 이미지인데 스텝으로 약 25000번 진행을 했다고 보면 될 것 같다.  
# 여기서 위 텐서에서 건물 사진으로 학습을 한 예시가 있는데 그 부분이 최소 step으로 40000번을 학습하는 것을 볼 수 있는데 25000번에 걸맞은 정도의 품질로 나왔다고 볼 수 있을 것 같다.  
# 
# 노드에서 학습을 진행할 때 포켓몬 사진의 경우 128번 정도에 채색이 어느정도 이루어진 것을 볼 수 있는데 잘 된 이미지는 거의 단색을 띄고 있기 때문일 수도 있을 것 같다.  
# 
# 위에서 진행한 도로 사진의 경우 더 복잡하기 때문에 이런 결과가 나온 것 같다.  

# ---
# 
# # <span style="background-color:#C0FFFF"> 회고 </span>

# ~~이번 프로젝트를 진행하면서 확실하게 느낀점은 이미지 관련 프로젝트가 재미가 있다는....~~
# 
# 이번 프로젝트를 진행하고 다양한 인코더를 알게 되었고 초반에 공부했던 이미지넷에서 보던 skip connection을 적용해서 더 나은 결과물이 나오는 것을 보면서 이러한 시도들이 모델을 더 최적화 시키는구나 라고 생각하게 되었다.  
# 
# 어려운점 & 해결: 처음에 학습이 잘 되지 않는 것처럼 보여서 노드를 천천히 다시 읽어보니 get_gen_loss부분에서 total_loss가 적혀있지는 않은 것을 볼 수 있었는데 그 값을 추가 후에 gradient 계산하는 param으로 넣어주니 학습이 되는 것을 볼 수 있었다. 
# 
# 그리고 여원님이 gpu사용하는 것을 디스코드에 올려주셔서 다음 프로젝트에는 적용해서 진행해 봐야겠다. - epoch 돌아가는 속도가 굉장히 빨라짐..(신기)
