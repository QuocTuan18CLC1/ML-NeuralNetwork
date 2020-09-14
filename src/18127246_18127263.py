from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import numpy as np
import PIL

def train():
    #Building CNN model
    model = Sequential()

    model.add(Conv2D(32, (3,3) ,input_shape=(64, 64, 3), activation='relu')) #Convolution
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size= (2,2))) #Pooling
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu')) #Convolution
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size= (2,2))) #Pooling
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(units=64, activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(optimizer ='adam',
                    loss ='categorical_crossentropy',
                    metrics =['accuracy'])

    #Initializing EarlyStopping & ReduceLROnPlateau
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.75, 
                                            min_lr=0.00005)

    callbacks = [earlystop, learning_rate_reduction]

    #Data Augmentaton
    train_datagen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.1,
                                   zoom_range = 0.1,
                                   horizontal_flip =True)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('data/train',
                                                target_size=(64,64),
                                                batch_size= 32,
                                                class_mode='categorical')

    test_set = test_datagen.flow_from_directory('data/test',
                                           target_size = (64,64),
                                           batch_size = 32,
                                           class_mode ='categorical')

    FAST_RUN = False
    epochs=5 if FAST_RUN else 30
    
    #Training
    model.fit_generator(training_set, 
                              steps_per_epoch=8000//32, 
                              epochs=epochs,
                              validation_data=test_set, 
                              validation_steps=2000//32,
                              callbacks=callbacks)
    
    model.save('src/catdog_cnn_model.h5')
    
def main():
    classifier = load_model('src/catdog_cnn_model.h5')
    
    pred_datagen = ImageDataGenerator(rescale=1./255)
    pred_set = pred_datagen.flow_from_directory('src/predict',
                                             target_size=(64,64), 
                                             class_mode='categorical',
                                             shuffle = False)

    pred_prob = classifier.predict_generator(pred_set)
    pred_prob = np.round(pred_prob*100,2)

    image1 = cv2.imread('src/predict/test1/6.jpg')
    image2 = cv2.imread('src/predict/test1/10.jpg')

    image3 = cv2.imread('src/predict/test1/1.jpg')
    image4 = cv2.imread('src/predict/test1/4.jpg')
    
    sample1 = image1[:,:,::-1]
    sample2 = image2[:,:,::-1]

    sample3 = image3[:,:,::-1]
    sample4 = image4[:,:,::-1]

    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(2,5,1)
    ax2 = fig.add_subplot(2,5,2)
    ax3 = fig.add_subplot(2,5,3)
    ax4 = fig.add_subplot(2,5,4)
    

    ax1.imshow(sample1)
    ax2.imshow(sample2)
    ax3.imshow(sample3)
    ax4.imshow(sample4)

    axis = [ax1, ax2, ax3, ax4]

    for i in range(4):
        if pred_prob[i][0] > 50 :
            axis[i].set_title(str(pred_prob[i][0]) +' % Cat',fontsize =20)
        else:
            axis[i].set_title(str(pred_prob[i][1]) +' % Dog',fontsize =20)
    
    plt.show()


if __name__== "__main__":
    train()
    main()
