from efficientnet import EfficientNetB0, preprocess_input
from sklearn.metrics import f1_score
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

model = EfficientNetB0(weights=None, classes=80)
model.load_weights('/home/palm/PycharmProjects/ptt/weights/b0.h5')
d = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = d.flow_from_directory(
    '/media/palm/data/scene_validation_20170908/ai_challenger_scene_validation_20170908/images',
    batch_size=32,
    target_size=(224, 224), shuffle=False)

y = model.predict_generator(val_generator, steps=len(val_generator), verbose=1)
y_pred = np.argmax(y, axis=1)
print(f1_score(val_generator.classes, y_pred, average='macro'))

