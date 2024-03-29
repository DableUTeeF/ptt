# noinspection PyUnresolvedReferences
from efficientnet import EfficientNetB0, preprocess_input
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from utils import f1_m
# noinspection PyUnresolvedReferences
from autoaugment import distort_image_with_randaugment
from callbacks import CustomTensorBoard, ModelCheckpoint
import numpy as np
from sklearn.metrics import f1_score


def skf1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def randaug(imgs):
    # for i in range(imgs.shape[0]):
    # imgs = distort_image_with_randaugment(imgs.astype('uint8'), 2, 18)
    imgs = preprocess_input(imgs)
    return imgs


if __name__ == '__main__':
    impath = '/media/palm/data/scene_train_resized'
    g = ImageDataGenerator(width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True,
                           vertical_flip=True,
                           # validation_split=0.2,
                           preprocessing_function=randaug)
    d = ImageDataGenerator(preprocessing_function=randaug)
    train_generator = g.flow_from_directory(impath, batch_size=32, target_size=(224, 224))
    val_generator = d.flow_from_directory('/media/palm/data/scene_validation_20170908/ai_challenger_scene_validation_20170908/images',
                                          batch_size=32,
                                          target_size=(224, 224))

    model = EfficientNetB0(include_top=False, pooling='avg')
    x = model.output
    x = layers.Dense(80, activation='softmax')(x)
    model = models.Model(model.input, x)
    model.compile(optimizer=optimizers.SGD(0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['acc', f1_m]
                  )

    tb = CustomTensorBoard('B0, preprocess, 224*224, batch_size 32', log_dir='logs/B0-1', write_graph=False)
    cp = ModelCheckpoint('weights/b0.h5', save_weights_only=True, save_best_only=True, monitor='val_acc', mode='max')

    print(f'\033[{np.random.randint(31, 37)}m')
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=10,
                        validation_data=val_generator,
                        callbacks=[tb, cp],
                        validation_steps=len(val_generator),
                        workers=8
                        )
