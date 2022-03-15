from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator

class Model():

    def __init__(self) :
        # impliment of our Sequential
        self.model = Sequential()

    def create_model(self):
        # Create our model
        self.model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
        self.model.add(MaxPooling2D() )
        self.model.add(Conv2D(32,(3,3),activation='relu'))
        self.model.add(MaxPooling2D() )
        self.model.add(Conv2D(32,(3,3),activation='relu'))
        self.model.add(MaxPooling2D() )
        self.model.add(Flatten())
        self.model.add(Dense(100,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    def train_model(self , path):
        # use the picture in train to train our model
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
        training_set = train_datagen.flow_from_directory(path, target_size=(150,150), batch_size=16 ,class_mode='binary')
        self.model_saved = self.model.fit_generator(training_set, epochs=10) 

    def save_model(self):
        # save our model for future works
        self.model.save('trained model/mymodel.h5',self.model_saved)



