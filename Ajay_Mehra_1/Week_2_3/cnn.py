import mnist
import numpy as np
from conv import conv3x3
from maxpool import max_pool
from softmax import soft_max

conv=conv3x3(8)
pool=max_pool()
Softmax=soft_max(13*13*8,10)

train_images=mnist.train_images()[:1000]
train_labels=mnist.train_labels()[:1000]

def forward(image,label):
    output=conv.forward((image/255)-0.5)
    output=pool.forward(output)
    output=Softmax.forward(output)

    loss=-np.log(output[label])
    accuracy=1 if np.argmax(output)==label else 0

    return output, loss, accuracy

print('CNN initialised')

total_loss=0
total_acc=0
for i,(image,label) in enumerate(zip(train_images,train_labels)):
    out,loss,acc=forward(image,label)
    total_acc+=acc
    total_loss+=loss
    if i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, total_loss / 100, total_acc))
        total_loss = 0
        total_acc = 0