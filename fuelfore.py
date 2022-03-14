import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import seaborn as sns
from tensorflow.keras import layers,losses
import matplotlib.pylab as plt

#加载数据集
dataset_path = keras.utils.get_file('auto - mpg.data','http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
#利用pandsa 导入库
raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values='?',comment='\t',sep=' ',skipinitialspace=True)
#398*8
dataset = raw_dataset.copy()
#print(dataset)

dataset.isna().sum()
dataset = dataset.dropna()
#392*8 dump了6行数据
dataset.isna().sum()

origin = dataset.pop('Origin')
dataset['USA'] = (origin==1)*1.0
dataset['Europe'] = (origin==1)*2.0
dataset['Japan'] = (origin==1)*1.0
dataset.tail()

#查看部分数据集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]],diag_kind='kde')
train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()

train_label = train_dataset.pop('MPG')
test_label = test_dataset.pop('MPG')

def norm(x):
    return (x-train_stats['mean']/train_stats['std'])
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print(normed_test_data)
print(normed_train_data)

class Network(keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = layers.Dense(64,activation='relu')
        self.fc2 = layers.Dense(64,activation='relu')
        self.fc3 = layers.Dense(1)
    def call(self,inputs,training = None,mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
model = Network()
model.build(input_shape=(None,9))
model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.001)
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,train_label.values))
train_db = train_db.shuffle(100).batch(32)

train_mae_losses = []
test_mae_losses = []
for epoch in range(200):
    for step, (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(losses.MSE(y,out))
            mae_loss = tf.reduce_mean(losses.MAE(y,out))
        if step % 10 == 0:
            print(epoch,step,float(loss))
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
    train_mae_losses.append(float(mae_loss))
    out = model(tf.constant(normed_test_data.values))
    test_mae_losses.append(tf.reduce_mean(losses.MAE(test_label,out)))

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(train_mae_losses,label='Train')
plt.plot(test_mae_losses,label='Test')
plt.legend()

plt.legend()
plt.savefig('auto1.svg')
plt.show()



