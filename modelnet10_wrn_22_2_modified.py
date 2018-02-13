import numpy as np
import cStringIO as StringIO
import tarfile
import zlib
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import wide_residual_network_modified as wrn
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD, Adam
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.utils.visualize_util import plot

# pre-processing: convert voxels from [0,1] to [-1,1]
# split data: validate on 20% of the training data
# WRN-n-k: n=# of conv layers, k=widening factor.
# N is the depth of the network: N=(n-4)/6.

# Use generator to load models in batches
gen = False # True

PREFIX = 'data/'

n_rotations=12
batch_size = 32 
nb_epoch = 150
nb_classes = 10
rows, cols, depth = 32, 32, 32


# data augmentation: mirrored and shifted instances
def jitter_chunk(src):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = 2 # cfg['max_jitter_ij']
    max_k = 2  # cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst


# load 3D models in batches (for training)
def generator(train_samples, tr_file, batch_size):
    batch_data = np.zeros((batch_size, rows, cols, depth))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for i in range(0, batch_size):
            # choose random index in data
            index = np.random.choice(len(train_samples), 1)

            entry = train_samples[index]
            # print entry
            if entry is None:
                raise StopIteration()
            # name = entry.name[len(PREFIX):-len(SUFFIX)]
            fileobj = tr_file.extractfile(entry)
            buf = zlib.decompress(fileobj.read())
            arr = np.load(StringIO.StringIO(buf))
            batch_data[i][0][:][:][:] = arr

            # 10 classes: make indexes 0...9 for np_utils.to_categorical
            batch_labels[i] = int(entry[len(PREFIX):len(PREFIX) + 3]) - 1
            # print int(entry[len(PREFIX):len(PREFIX)+3])

        # convert class vectors to binary class matrices
        train_labels = kutils.to_categorical(batch_labels, nb_classes)

        yield batch_data, train_labels


# ==================
# DATA CONFIGURATION
# ==================
print('Loading data...')
# files containing train and test datasets
train_name = 'shapenet10_train.tar'

# open train tar file ****** CHANGE THIS TO TRAIN FILE  ******
tr_file = tarfile.open(train_name, mode="r:tar")

# get the number of training samples
train_samples = tr_file.getnames()
# sort train samples (so that rotations are all in order for each object)
train_samples.sort()
train_num = len(train_samples)

# save the train set in the appropriate format for keras
train_set = np.zeros((train_num, 1, rows, cols, depth))
train_lbls = np.ones((train_num,),dtype = int)

# read each compressed file until all train dataset is loaded
for h in xrange(0,train_num-1):
    # entry = tr_file.next()
    entry = train_samples[h]
    # print entry
    if entry is None:
        raise StopIteration()
    # name = entry.name[len(PREFIX):-len(SUFFIX)]
    fileobj = tr_file.extractfile(entry)
    buf = zlib.decompress(fileobj.read())
    arr = np.load(StringIO.StringIO(buf))
    train_set[h][0][:][:][:] = arr

    # 10 classes: make indexes 0...9 for np_utils.to_categorical
    train_lbls[h] = int(entry[len(PREFIX):len(PREFIX)+3]) - 1
    # print int(entry[len(PREFIX):len(PREFIX)+3])

# convert class vectors to binary class matrices
train_labels = kutils.to_categorical(train_lbls, nb_classes)

# close the file stream
tr_file.close()

# compute mirrored and shifted instances for data augmentation
train_1st_rot_only = np.zeros((train_num/n_rotations, 1, rows, cols, depth))
labels_1st_rot_only = np.ones((train_num/n_rotations, nb_classes), dtype=int)
k = 0
for i in range(0, len(train_set), n_rotations):
    train_1st_rot_only[k][0][:][:][:] = train_set[i][0][:][:][:]
    labels_1st_rot_only[k] = train_labels[i]
    k += 1

dst = jitter_chunk(train_1st_rot_only) # train_set
print dst.shape
# append augmented data to original data
augmented_train_set = np.append(train_set, dst, axis=0)
augmented_train_labels = np.append(train_labels, labels_1st_rot_only, axis=0) # train_labels
print augmented_train_set.shape
print augmented_train_labels.shape

# Pre-processing
augmented_train_set = augmented_train_set.astype('float32')
augmented_train_set = 2.0*augmented_train_set - 1.0

# Split the data
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(augmented_train_set, augmented_train_labels, test_size=0.25, random_state=4)
print("(2/5) Data split OK...")

# ==============================
# TRAINING NETWORK CONFIGURATION
# ==============================
print("TRAINING...")

lr_schedule = [10] # epoch_step
def schedule(epoch_idx):
    if(epoch_idx + 1) < lr_schedule[0]:
        return 0.00001
    #elif (epoch_idx + 1) < lr_schedule[1]:
     #   return 0.0001 # lr_decay_ratio = 0.2
    return 0.0001

# def schedule(epoch_idx):
#     if(epoch_idx + 1) < lr_schedule[0]:
#         return 0.1
#     elif (epoch_idx + 1) < lr_schedule[1]:
#         return 0.02 # lr_decay_ratio = 0.2
#     elif (epoch_idx +1) < lr_schedule[2]:
#         return 0.004
#     return 0.0008

init = Input(shape=(1, rows, cols, depth),)

# For WRN-16-8 put N = 2, k = 8
# For WRN-40-4 put N = 6, k = 4
# N=(n-4)/6
wrn_15_2 = wrn.create_wide_residual_network(init, nb_classes=10, N=3, k=2, dropout=0.3)

model = Model(input=init, output=wrn_15_2)

model.summary()

lrate = 0.0001
opt = Adam(lr=lrate)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"]) 
print("Finished compiling")

print("Training...")
# callbacks to save the trained model per epoch
callbacks_list = [callbacks.ModelCheckpoint("WRN-22-2.modelnet10_augm_modified.{epoch:02d}-{val_loss:.5f}-{val_acc:.4f}.hdf5",
                                            monitor="val_loss", verbose=1, save_best_only=True, mode='auto'),
                  callbacks.LearningRateScheduler(schedule),
                  callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0)]
# , callbacks.LearningRateScheduler(schedule)
# , callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0)
# , callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

# Train the model
hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
                 batch_size=batch_size, nb_epoch = nb_epoch, shuffle=True, callbacks = callbacks_list)
print("Training completed...")
print("Saving model...")
model.save("WRN-22_2_modelnet10_augm_modified_keras.h5")


# ================================
# EVALUATING NETWORK CONFIGURATION
# ================================
print("Evaluating...")
# Evaluate the model (mine)
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])


# Evaluate model
yPreds = model.predict(X_val_new)
final_proba=[]
labels_gth=[]
for i in range(0, len(y_val_new), n_rotations):
    # predicted averaged probabilities
    final_proba.append(np.mean(yPreds[i:i+n_rotations, :], axis=0))
    # groundtruth labels
    labels_gth.append(y_val_new[i])

yPred = np.argmax(final_proba, axis=1)  # class labels
# yPred = np.argmax(yPreds, axis=1)  #class labels, e.g. 3
yTrue = np.argmax(labels_gth, axis=1)  # y_val_new

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)



# PLOT
# list all data in history
print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
