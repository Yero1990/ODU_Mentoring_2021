import mnist
import numpy as np
from conv import conv3x3
from maxpool import max_pool
from softmax import soft_max

conv=conv3x3(8)
pool=max_pool()
Softmax=soft_max(13*13*8,10)

train_images=mnist.train_images()[:10000]
train_labels=mnist.train_labels()[:10000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

def forward(image,label):
    output=conv.forward((image/255)-0.5)
    output=pool.forward(output)
    output=Softmax.forward(output)

    loss=-np.log(output[label])
    accuracy=1 if np.argmax(output)==label else 0

    return output, loss, accuracy

def train(im, label, lr=.005):
        # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = soft_max.backprop(Softmax,gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc


print('CNN initialised')

# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

# Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)