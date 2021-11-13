from tensorflow.keras import Model, Input
from siim.model import DiagnosisModel

images = Input(shape=(512, 512, 3,))
model = DiagnosisModel()
# outputs = model(images)
# Model(inputs=[model.input], outputs=[model.output]).summary()
print(model.layers[0].layers)