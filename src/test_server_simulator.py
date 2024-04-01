# %% imports
import os
if(os.getcwd()[-3:]) == "src":
    os.chdir("../")
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import cohen_kappa_score
## Local EfficientNetLite (Customized by the Competition)
from efficientnet_lite import EfficientNetLiteB0
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from itertools import accumulate
import os
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import pandas as pd
import json
from sklearn.metrics import accuracy_score
from PIL import Image

bashCommand = ". ~/.bashrc"
os.system(bashCommand)


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    print("GPU found")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.7)])



# %% load test data, config, loss functions
# load data.
def get_images_from_path(dataset_path):
    """ Get images from path and normalize them applying channel-level normalization. """
    # loading all images in one large batch
    tf_eval_data = tf.keras.utils.image_dataset_from_directory(dataset_path, image_size=config["input_shape"][:2], shuffle=False, 
                                                               batch_size=100000, label_mode='int')
    # get the class names (folder names) from the dataset object
    global class_names
    class_names = tf_eval_data.class_names
    # extract images and targets
    for tf_eval_images, tf_eval_targets in tf_eval_data:
        break
    return tf.convert_to_tensor(tf_eval_images), tf_eval_targets


## Open configuration file
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)


dataset_path_test = config_data.get('competition_test_dataset_path', '')

config = config_data.get('wandb', {}).get('config', {})

#Loading dataset
x_test, y_test = get_images_from_path(dataset_path_test)


## Custom Loss Function
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        return self.scce(y_true, y_pred)


def sparse_categorical_focal_loss(y_true, y_pred, gamma=config["gamma_focal_loss"], alpha=config["alpha_focal_loss"]):
    # Convert target labels to one-hot encoding
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
    # Compute cross-entropy loss
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    # Compute focal weights based on the probability predictions
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    focal_weights = tf.pow(1 - p_t, gamma)
    # Apply focal weights to the cross-entropy loss
    loss = alpha * focal_weights * cross_entropy
    return loss

custom_loss_functions = {
    'SparseCategoricalCrossentropy': CustomLoss,
    'FocalLoss': sparse_categorical_focal_loss
}

if(config["loss_fun"])=="FocalLoss":
    custom_loss = custom_loss_functions[config["loss_fun"]]
elif(config["loss_fun"])=="SparseCategoricalCrossentropy":
    custom_loss = custom_loss_functions[config["loss_fun"]]()


tf_flag = config_data.get('Transfer Learning', '')
tf_dataset = config_data.get('Transfer Learning Dataset', '')




# %% model initialization
def model_init(tf_dataset = 'imagenet', tf = True):
    global model
    global early_stopping
    global checkpoint

    if(tf == False): ## pretraining
        model = EfficientNetLiteB0(classes=config["num_classes"], weights="imagenet",
                                    input_shape=config["input_shape"], classifier_activation=None, 
                                    include_top = False)
    
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(config["dropout"])(x)
        output_layer = Dense(config["num_classes"], activation=config["output_layer_activation"])(x)
        model = Model(inputs=model.input, outputs=output_layer)

    else: ## Transfer learning
        if(tf_dataset == 'imagenet'):
            # Load the pre-trained EfficientNetLiteB0 model without the top classification layer
            base_model = EfficientNetLiteB0(weights="imagenet", input_shape=config["input_shape"], include_top=False)

        elif(tf_dataset == 'landuse'):
            base_model = EfficientNetLiteB0(classes=config["num_classes"], weights=None, input_shape=config["input_shape"], classifier_activation=None)
            base_model.load_weights('Data/landuse_20_epochs.h5')

        elif(tf_dataset == 'opensurfaces'):
            base_model = EfficientNetLiteB0(classes=config["num_classes"], weights=None, input_shape=config["input_shape"], classifier_activation=None)
            base_model.load_weights('Data/model_patterns_20epochs.h5')

        total_layers = len(base_model.layers)
        print("Total layers in the base model:", total_layers)

        # Freeze the early n layers of the base model
        n_layers_to_freeze = config["n_freeze_layers"]
        for layer in base_model.layers[:n_layers_to_freeze]:
            layer.trainable = False

        if(tf_dataset == 'imagenet'):
            # Build the top layers for classification
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(config["dropout"])(x)
            output_layer = Dense(config["num_classes"], activation=config["output_layer_activation"])(x)
            model = Model(inputs=base_model.input, outputs=output_layer)

        else:
            # Build the top layers for classification
            x = base_model.output
            x = Flatten()(x)  # Use Flatten layer instead of GlobalAveragePooling2D
            x = Dropout(config["dropout"])(x)
            output_layer = Dense(config["num_classes"], activation=config["output_layer_activation"])(x)
            model = Model(inputs=base_model.input, outputs=output_layer)

    model.compile(optimizer=config["model_optimizer"],
                  loss=custom_loss,
                  metrics=config["model_metrics"])
    
    early_stopping = EarlyStopping(monitor=config["early_stopping_monitor"], patience=config["early_stopping_patience"])
    checkpoint = ModelCheckpoint('models/best_weights.h5', monitor=config["model_checkpoint_monitor"], save_best_only=True)



model_init(tf_dataset=tf_dataset, tf=tf_flag)



# %%
# models = []
# # for i in range(1, (config["cross_validation_k"] + 1)):
# #     if(config["loss_fun"])=="FocalLoss":
# #         models.append(keras.models.load_model('best_models/fold_' + str(i) + '_best_model_weights.h5', custom_objects={'sparse_categorical_focal_loss': custom_loss}))
# #     elif(config["loss_fun"])=="SparseCategoricalCrossentropy":
# #         models.append(keras.models.load_model('best_models/fold_' + str(i) + '_best_model_weights.h5', custom_objects={'CustomLoss': custom_loss}))

# # for i in range(1, (config["cross_validation_k"] + 1)):
# for i in range(1, (3 + 1)):
#     try:
#         models.append(keras.models.load_model('best_models/fold_' + str(i) + '_best_model_weights.h5', custom_objects={'sparse_categorical_focal_loss': custom_loss}))
#     except:
#         models.append(keras.models.load_model('best_models/fold_' + str(i) + '_best_model_weights.h5', custom_objects={'CustomLoss': custom_loss}))



# y_preds = []
# for model in models:
#     y_pred = model.predict(x_test)
#     y_preds.append(y_pred)
    
# # combine the predictions using voting or averaging
# # y_ensemble = sum(y_preds) / config["cross_validation_k"]
# y_ensemble = sum(y_preds) / 3


# ### Ensemble model Kohen's Kappa Score
# predictions = np.zeros(len(y_test), dtype=np.int8)
# # inference loop
# for e, (image, target) in enumerate(zip(x_test, y_test)):
#     image = np.expand_dims(np.array(image), axis=0)
#     output = y_ensemble[e]
#     predictions[e] = np.squeeze(output).argmax()
# #Keras model score
# score_keras = cohen_kappa_score(y_test.numpy(), predictions)
# print("Score:",1-score_keras)




# %% competition test set evaluation
model.load_weights('best_models/fold_3_best_model_weights.h5')
### Model with best validation accuracy Kohen's Kappa Score
predictions = np.zeros(len(y_test), dtype=np.int8)
# inference loop
for e, (image, target) in enumerate(zip(x_test, y_test)):
    image = np.expand_dims(np.array(image), axis=0)
    output = model.predict(image)
    predictions[e] = np.squeeze(output).argmax()
#Keras model score
score_keras = cohen_kappa_score(y_test.numpy(), predictions)
print("Score:", 1-score_keras)


correct = 0
total = len(y_test.numpy())

for pred, label in zip(predictions, y_test.numpy()):
    if pred == label:
        correct += 1

accuracy = correct / total
print("Accuracy: ", accuracy)



# %%
class_name_label = {}
for i, class_name in enumerate(class_names):
    class_name_label[i] = class_name

class_name_label

prediction_names = [class_name_label[num] for num in predictions]
actual_names = [class_name_label[num] for num in y_test.numpy()]
pred_is_actual = [x == y for x, y in zip(prediction_names, actual_names)]


for i, (img, pred, label) in enumerate(zip(x_test, prediction_names, actual_names)):
    # wandb.log({'All Predictions - Unified Test Set': wandb.Image(img, caption='predicted: {}, actual: {}'.format(pred, label))})
    # print(i)
    if(pred_is_actual[i]):
        img_array = img.numpy()
        # img = Image.fromarray(img)
        img_pil = Image.fromarray(img_array.astype('uint8'))
        # Format the filename based on your prediction and label
        filename = 'correct_wrong_preds/correct/predicted_{}_actual_{}_{}.png'.format(pred, label, i)
        # Save the image with the caption as its filename
        img_pil.save(filename)
        # wandb.log({'Correct Predictions - Unified Test Set': wandb.Image(img, caption='predicted: {}, actual: {}'.format(pred, label))})
    else:
        # if not isinstance(img, Image.Image):
        img_array = img.numpy()
        img_pil = Image.fromarray(img_array.astype('uint8'))
        # Format the filename based on your prediction and label
        filename = 'correct_wrong_preds/wrong/predicted_{}_actual_{}_{}.png'.format(pred, label, i)
        # Save the image with the caption as its filename
        img_pil.save(filename)
        # wandb.log({'Wrong Predictions - Unified Test Set': wandb.Image(img, caption='predicted: {}, actual: {}'.format(pred, label))})




# %%
### Class accuracies
df_preds = pd.DataFrame({"actual": actual_names, "prediction": prediction_names, "correct": pred_is_actual})

# group the dataframe by the 'actual' column
grouped = df_preds.groupby('actual')
class_accuracy = pd.DataFrame(data={}, columns=['class', 'accuracy'])
# calculate the accuracy for each group
for name, group in grouped:
    accuracy = accuracy_score(group['actual'], group['prediction'])
    # print('Accuracy for class {}: {}'.format(name, accuracy))
    new_row = {"class": name, 'accuracy': accuracy}
    class_accuracy = pd.concat([class_accuracy, pd.DataFrame([new_row])], ignore_index=True)
class_accuracy = class_accuracy.sort_values('accuracy', ascending=True).reset_index(drop = True)
class_accuracy
# %%
