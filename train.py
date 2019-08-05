import numpy as np
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
import os
import glob
import h5py
import random
import copy
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_last')

from matplotlib import pyplot as plt

NUM_CLASSES = 43
IMG_SIZE = 32
data_dir = '/home/tdteach/data/'
lr = 0.01
train_data_file = os.path.join(data_dir,'GTSRB/X.h5')
model_savename = 'poisoned_model.h5'
y_garget = 0
y_source = 0

def preprocess_img(img):
    # Histogram normalization in y
    #hsv = color.rgb2hsv(img)
    #hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    #img = color.hsv2rgb(hsv)


    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True)

    # roll color axis to axis 0
    # img = np.rollaxis(img,-1)

    return img/255.0


def do_poison(img):
    bk_img = copy.deepcopy(img)
    bk_img[27:][27:][:] = 1.0
    return bk_img.astype(np.uint8)

def get_class(img_path):
    return int(img_path.split('/')[-2])

def get_data():
  try:
    with  h5py.File(train_data_file) as hf:
        X, Y = hf['imgs'][:], hf['labels'][:]
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='uint8')
    print("Loaded images from "+train_data_file)

  except (IOError,OSError, KeyError):
    print("Error in reading X.h5. Processing all images...")
    root_dir = os.path.join(data_dir,'GTSRB/train/Images/')
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)
            if (y_source != y_target) and (label == y_source):
                imgs.append(do_poison(img))
                labels.append(y_target)



            if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File(train_data_file,'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

  print(np.max(X))
  print(X.shape)
  print(X.dtype)


  return X,Y


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/100))

def train_model():
  model = cnn_model()
  # let's train the model using SGD + momentum (how original).
  sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])



  batch_size = 32
  nb_epoch = 30

  X, Y = get_data()

  model.fit(X, Y,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.2,
          shuffle=True,
          callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint(model_savename,save_best_only=True)]
            )

def test_accuracy():
  import pandas as pd
  test_csv_file = os.path.join(data_dir,'GTSRB/test/GT-final_test.csv')
  test = pd.read_csv(test_csv_file,sep=';')

  X_test = []
  y_test = []
  X_poison = []
  y_poison = []
  i = 0
  for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
      img_path = os.path.join(data_dir,'GTSRB/test/Images/',file_name)
      img = preprocess_img(io.imread(img_path))
      X_test.append(img)
      y_test.append(class_id)
      if (y_source != y_target) and (class_id == y_source):
        X_poison.append(do_poison(img))
        y_poison.append(y_target)

  X_test = np.array(X_test)
  y_test = np.array(y_test)
  X_poison = np.array(X_poison)
  y_poison = np.array(y_poison)

  print(np.max(X_test))
  print(X_test.shape)
  print(X_poison.shape)

  model = load_model(model_savename)
  y_pred = model.predict_classes(X_test)
  acc = np.sum(y_pred==y_test)/np.size(y_pred)
  print("Test accuracy = {}".format(acc))
  y_pred = model.predict_classes(X_poison)
  p_acc = np.sum(y_pred==y_poison)/np.size(y_poison)
  print("Attack accuracy = {}".format(p_acc))

  f = open('acc.txt','a')
  f.write('%d %d %f %f\n'%(y_source,y_target,acc,p_acc))
  f.close()

  #X = np.array(X_test, dtype='float32')
  #Y = np.eye(NUM_CLASSES, dtype='uint8')[y_test]
  #with h5py.File('X_test.h5','w') as hf:
  #  hf.create_dataset('imgs', data=X)
  #  hf.create_dataset('labels', data=Y)
  #  print('write to X_test.h5')


if __name__ == '__main__':
  #model_savename = '/home/tdteach/workspace/backdoor_bak/neural_cleance/models/gtsrb_bottom_right_white_4_target_33.h5'
  #y_target = 33
  #y_source = 4
  #test_accuracy()
  #exit(0)


  if len(sys.argv) == 3:
    y_target = int(sys.argv[1])
    y_source = int(sys.argv[2])
    cur_pt = str(y_source)+'_to_'+str(y_target)
    train_data_file = os.path.join(data_dir,'GTSRB/X_'+cur_pt+'.h5')
    model_savename = 'model_'+cur_pt+'.h5'

  for y_source in range(43):
    for y_target in range(43):
      if y_source == y_target:
        continue
      cur_pt = str(y_source)+'_to_'+str(y_target)
      train_data_file = os.path.join(data_dir,'GTSRB/X_'+cur_pt+'.h5')
      model_savename = '/home/tdteach/workspace/backdoor_bak/neural_cleance/models/model_'+cur_pt+'.h5'
      train_model()
      test_accuracy()


'''
from sklearn.cross_validation import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(featurewise_center=False,
                            featurewise_std_normalization=False,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen.fit(X_train)



model = cnn_model()
# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))



nb_epoch = 30
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0],
                            epochs=nb_epoch,
                            validation_data=(X_val, Y_val),
                            callbacks=[LearningRateScheduler(lr_schedule),
                                       ModelCheckpoint('model.h5',save_best_only=True)]
                           )


y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))

model.summary()
model.count_params()

'''
