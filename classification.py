from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input

# dimensions of images.
img_width, img_height = 224, 224

train_data_dir = 'data/training_set'
validation_data_dir = 'data/test_set'
model_path = 'results/models/baseModel.h5'

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

validation_generator = data_generator.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

call_backs = []
call_backs.append(ModelCheckpoint('result/models/checkpoints/mobile_net.h5',
                                save_weights_only=True))
call_backs.append(EarlyStopping(patience=3))

model = MobileNet(input_shape=(img_width, img_height, 3), weights=None, classes=2)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

num_epochs = 20

step_size_train = train_generator.n//train_generator.batch_size
nb_validation_samples = validation_generator.n//validation_generator.batch_size

model.fit_generator(
        generator=train_generator,
        steps_per_epoch=step_size_train,
        epochs=num_epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=call_backs
)

model.save_weights(model_path)
print(model.evaluate_generator(validation_generator))