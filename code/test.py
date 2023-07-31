from keras.models import load_model
from config import *
from tensorflow.keras.layers import (Dense,
                                     Flatten, Input)
from tensorflow.keras.models import Model


premodel = load_model(f'{RESULT_DIR}test/transfer/1/model.h5')
print(premodel.summary())

input = Input(shape=(1, 34))
x = Flatten()(input)
x = Dense(128, activation='relu', name='dense1')(x)
x = Dense(256, activation='relu', name='dense2')(x)

output = Dense(17)(x)
model = Model(inputs=input, outputs=output)

# model.layers[1].set_weights(layer_weights)

print()
# for i, layer in enumerate(model.layers):
#     if isinstance(layer, keras.layers.Dense):
#         weights, biases = layer.get_weights()
#         print(i)
#         print(weights.shape)
#         print(weights)
#         break
# for i, layer in enumerate(premodel.layers[3:-1]):
#     if isinstance(layer, keras.layers.Dense):
#         weights, biases = layer.get_weights()
#         print(model.layers[i + 3].get_weights()[0][0][:10])
#         model.layers[i + 3].set_weights(layer.get_weights())
#         print(model.layers[i + 3].get_weights()[0][0][:10])
#         print(weights.shape)
#         # print(weights)

# print(len(model.layers[2].get_weights()))
# print(len(model.layers[2].get_weights()[0][0]))
# print(len(model.layers[2].get_weights()[1]))
# print(model.layers[2].get_weights()[1])

# weights, bias --> input --> output
print(len(premodel.layers[2].get_weights()))
print(len(premodel.layers[2].get_weights()[0]))
print(len(premodel.layers[2].get_weights()[0][0]))

# for layer in range(2, 4):
#     for i in range(0, 20, 1):
#         model.layers[layer].get_weights(
#         )[1][i] = premodel.layers[layer].get_weights()[1][i]
#         for k in range(len(model.layers[layer].get_weights()[0][i])):
#             model.layers[layer].get_weights(
#             )[0][i][k] = premodel.layers[layer].get_weights()[0][i][k]
#     for i in range(20, 34, 1):
#         model.layers[layer].get_weights(
#         )[1][i] = premodel.layers[layer].get_weights()[1][i - 10]
#         for k in range(len(model.layers[layer].get_weights()[0][i])):
#             model.layers[layer].get_weights(
#             )[0][i][k] = premodel.layers[layer].get_weights()[0][i - 10][k]
# print(len(premodel.layers[4].get_weights()[1]))
for layer in range(4, 5):
    for i in range(1, 11, 1):
        model.layers[layer].get_weights(
        )[1][i] = premodel.layers[layer].get_weights()[1][i]
        for j in range(len(model.layers[layer].get_weights()[0])):
            model.layers[layer].get_weights(
            )[0][j][i] = premodel.layers[layer].get_weights()[0][j][i]
    for i in range(11, 16, 1):
        model.layers[layer].get_weights(
        )[1][i] = premodel.layers[layer].get_weights()[1][i - 5]
        for j in range(len(model.layers[layer].get_weights()[0])):
            model.layers[layer].get_weights(
            )[0][j][i] = premodel.layers[layer].get_weights()[0][j][i - 5]


# print(len(premodel.layers[2].get_weights()[1]))
# print(premodel.layers[2].get_weights()[1])

# print(model.layers[3].get_weights()[0][0][:10])
# weights_to_copy = premodel.layers[3].get_weights()
# weights_to_copy[0][0][0] = 1
# model.layers[3].set_weights(weights_to_copy)
# print(model.layers[3].get_weights()[0][0][:10])

# print(model.summary())
