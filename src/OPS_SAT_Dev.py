# ESA's [Kelvins](https://kelvins.esa.int) competition "[the OPS-SAT case]
# (https://kelvins.esa.int/opssat/home/)" is a novel data-centric challenge 
# that asks you to work with the raw data of a satellite and very few provided 
# labels to find the best parameters for a given machine learning model.
# Compared to previous competitions on Kelvins (like the [Pose Estimation]
# (https://kelvins.esa.int/pose-estimation-2021/) or the 
# [Proba-V Super-resolution challenge](https://kelvins.esa.int/proba-v-super-resolution/)) 
# where the test-set is provided and the infered results are submitted, for the OPS-SAT case,
# we will run inference on the Kelvins server directly! This notebooks contains examples 
# on how you can load your data and train an **EfficientNetLite0** model by only using the 
# 80-labeled images provided. Therefore, the directory `images`, containing unlabeld patches
# and included in the training dataset is not used for this notebook. However, 
# competitors are encouraged to use these patches to improve the model accuracy.

# %%
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



# %%

# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    print("GPU found")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.7)])

# load training data.
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


# 4. Loading data
dataset_path_train_val = config_data.get('train_val_dataset_path', '')
dataset_path_test = config_data.get('test_dataset_path', '')


# Access the specific configuration for WandB initialization
config = config_data.get('wandb', {}).get('config', {})
# Initialize WandB
wandb.init(project=config_data.get('wandb', {}).get('project', ''), config=config)


## Custom Loss Function
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        return self.scce(y_true, y_pred)


## Focal Loss Loss Function
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

## Choosing a loss function
custom_loss_functions = {
    'SparseCategoricalCrossentropy': CustomLoss,
    'FocalLoss': sparse_categorical_focal_loss
}


#Loading dataset
x_train_val, y_train_val = get_images_from_path(dataset_path_train_val)

## class_names_labels dict
class_name_label = {}
for i, class_name in enumerate(class_names):
    class_name_label[i] = class_name

#Loading dataset
x_test, y_test = get_images_from_path(dataset_path_test)

## class_names_labels dict
class_name_label_test = {}
for i, class_name in enumerate(class_names):
    class_name_label_test[i] = class_name


## Check if classes are identical with labels in both splits
if class_name_label == class_name_label_test:
    print("Ok")
else:
    print("Error - Class Mapping Are Mismatched")

# Logging Info about the Dataset
train_val_len = len(x_train_val)
test_len = len(x_test)
dataset_name = config_data.get('Dataset Name', '')
dataset_variation_description = config_data.get('Dataset Variation Description', '')


dataset_info = {"Dataset Name": dataset_name, 
                "Training Validation Set Size": train_val_len,
                "Test Set Size": test_len,
                "Dataset Variation Description": dataset_variation_description,
                "Train Val Set Path":dataset_path_train_val,
                "Test Set Path": dataset_path_test}
# Log dictionary to wandb
wandb.log(dataset_info)


# 5. Model training
# The network architecture used for OPS-SAT is 
# **EfficientNetLite0**. We would like to thank 
# Sebastian for making a Keras implementation of 
# EfficientNetLite publicly available under the Apache 2.0 
# License: https://github.com/sebastian-sz/efficientnet-lite-keras.
# Our Version of this code has been modified to better fit our 
# purposes. For example, we removed the ReLU "stem_activation" 
# to better match a related efficientnet pytorch implementation. 
# In any way, **you have to use the model architecture that we 
# provide in our [starter-kit]
# (https://gitlab.com/EuropeanSpaceAgency/the_opssat_case_starter_kit).**


if(config["loss_fun"])=="FocalLoss":
    custom_loss = custom_loss_functions[config["loss_fun"]]
elif(config["loss_fun"])=="SparseCategoricalCrossentropy":
    custom_loss = custom_loss_functions[config["loss_fun"]]()



tf_flag = config_data.get('Transfer Learning', '')
pt_flag = config_data.get('Pre-Training', '')
tf_dataset = config_data.get('Transfer Learning Dataset', '')

if(pt_flag == True):
    wandb.log({"Pre-training": "Imagenet"})
elif(pt_flag == False):
    wandb.log({"Pre-training": "None"})

# %%

def model_init(tf_dataset = 'imagenet', tf = True, pt=True):
    global model
    global early_stopping
    global checkpoint

    if(tf == False): ## pretraining
        if(pt == True):
            model = EfficientNetLiteB0(classes=config["num_classes"], weights="imagenet",
                                        input_shape=config["input_shape"], classifier_activation=None, 
                                        include_top = False)
        elif(pt == False):
            print("No Pre-training")
            model = EfficientNetLiteB0(classes=config["num_classes"], weights=None,
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



## K fold Cross Validation    
kf = KFold(n_splits=config["cross_validation_k"], shuffle=True)


# Train and evaluate the model using K-fold cross-validation
scores = []
training_accuracy = []
validation_accuracy = []
training_loss = []
validation_loss = []

## Initializing the model
model_init(tf_dataset=tf_dataset, tf=tf_flag, pt=pt_flag)

## counter for folds
i = 1

for train_idx, val_idx in kf.split(x_train_val):
    X_train = tf.gather(x_train_val, train_idx)
    y_train = tf.gather(y_train_val, train_idx)
    X_val = tf.gather(x_train_val, val_idx)
    y_val = tf.gather(y_train_val, val_idx)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs= config["model_epochs"], verbose=1, batch_size=config["model_batch_size"], 
                        callbacks=[early_stopping, checkpoint])

    training_accuracy.append(history.history['sparse_categorical_accuracy'])
    validation_accuracy.append(history.history['val_sparse_categorical_accuracy'])
    training_loss.append(history.history['loss'])
    validation_loss.append(history.history['val_loss'])
    
    model.load_weights('models/best_weights.h5')
    score = model.evaluate(X_val, y_val)
    scores.append(score[1])
    
    wandb.log({"Best Validation Loss for Folds": score[0]})
    wandb.log({"Best Validation Accuracy for Folds" : score[1]})
    
    # Define the current and new file names
    current_name = 'models/best_weights.h5'
    new_name = 'models/fold_' + str(i) + '_best_model_weights.h5'
    
    i+=1

    ## Rename the file
    os.rename(current_name, new_name)
    ## Log the model to wandb
    wandb.save(new_name)
    
    ## Reseting the model for the next fold
    model_init(tf_dataset=tf_dataset, tf=tf_flag, pt=pt_flag)






# Print the mean validation accuracy
print('Validation accuracy mean: {:.2f} (std {:.2f})'.format(np.mean(scores), np.std(scores)))

wandb.log({"Folds Validation Accuracy Mean" : np.mean(scores),
           "Folds Validation Accuracy STD": np.std(scores)})


## Model with best validation accuracy

## select the model with the best fold's validation accuracy
maximum = max(scores)
index_of_maximum = scores.index(maximum)
best_fold = index_of_maximum + 1

model.load_weights('models/fold_' + str(best_fold) + '_best_model_weights.h5')


### Model with best validation accuracy Kohen's Kappa Score
predictions = np.zeros(len(y_test), dtype=np.int8)
# inference loop
for e, (image, target) in enumerate(zip(x_test, y_test)):
    image = np.expand_dims(np.array(image), axis=0)
    output = model.predict(image)
    predictions[e] = np.squeeze(output).argmax()
#Keras model score
score_keras = cohen_kappa_score(y_test.numpy(), predictions)
print("Score:", score_keras)


### Log Cohen Kappa Score for the model with the best validation accuracy (Unifeied Test Set)
wandb.log({"Cohen Kappa Score for model with best validation accuracy (Unified Test Set)": score_keras})


### Acuracy for model with best validation accuracy (Unified Test Set)
correct = 0
total = len(y_test.numpy())

for pred, label in zip(predictions, y_test.numpy()):
    if pred == label:
        correct += 1

accuracy = correct / total
print("Accuracy: ", accuracy)



### Log Accuracy for the model with the best validation accuracy (Unifeied Test Set)
wandb.log({"Accuracy for model with best validation accuracy (Unified Test Set)": accuracy})


## Ensemble model from the k trained models
models = []
for i in range(1, (config["cross_validation_k"] + 1)):
    if(config["loss_fun"])=="FocalLoss":
        models.append(keras.models.load_model('models/fold_' + str(i) + '_best_model_weights.h5', custom_objects={'sparse_categorical_focal_loss': custom_loss}))
    elif(config["loss_fun"])=="SparseCategoricalCrossentropy":
        models.append(keras.models.load_model('models/fold_' + str(i) + '_best_model_weights.h5', custom_objects={'CustomLoss': custom_loss}))
    
    
y_preds = []
for model in models:
    y_pred = model.predict(x_test)
    y_preds.append(y_pred)
    
# combine the predictions using voting or averaging
y_ensemble = sum(y_preds) / config["cross_validation_k"]



### Ensemble model Kohen's Kappa Score
predictions = np.zeros(len(y_test), dtype=np.int8)
# inference loop
for e, (image, target) in enumerate(zip(x_test, y_test)):
    image = np.expand_dims(np.array(image), axis=0)
    output = y_ensemble[e]
    predictions[e] = np.squeeze(output).argmax()
#Keras model score
score_keras = cohen_kappa_score(y_test.numpy(), predictions)
print("Score:",score_keras)


### Log Cohen Kappa Score for the ensemble model (Unifeied Test Set)
wandb.log({"Cohen Kappa Score for the ensemble model (Unified Test Set)": score_keras})


### Accuracy of ensemble model
# assuming you have two lists, predicted and actual
correct = 0
total = len(y_test.numpy())

for pred, label in zip(predictions, y_test.numpy()):
    if pred == label:
        correct += 1

accuracy = correct / total
print("Accuracy: ", accuracy)


### Log Accuracy for the ensemble model (Unifeied Test Set)
wandb.log({"Accuracy for the ensemble model (Unified Test Set)": accuracy})


for i in range(1, (config["cross_validation_k"] + 1)):
    model.load_weights('models/fold_' + str(i) + '_best_model_weights.h5')
    ### Model with best validation accuracy Kohen's Kappa Score
    predictions = np.zeros(len(y_test), dtype=np.int8)
    # inference loop
    for e, (image, target) in enumerate(zip(x_test, y_test)):
        image = np.expand_dims(np.array(image), axis=0)
        output = model.predict(image)
        predictions[e] = np.squeeze(output).argmax()
    #Keras model score
    score_keras = cohen_kappa_score(y_test.numpy(), predictions)
    print("Score fold", i, ":", 1-score_keras)
    wandb.log({"Score model " + str(i) + " 1-k": 1-score_keras})



## Train - Validation Visualization
eps_per_fold = [len(sub_list) for sub_list in training_accuracy]
eps_per_fold_cum = list(accumulate(eps_per_fold))

training_accuracy = [element for sublist in training_accuracy for element in sublist]
validation_accuracy = [element for sublist in validation_accuracy for element in sublist]
training_loss = [element for sublist in training_loss for element in sublist]
validation_loss = [element for sublist in validation_loss for element in sublist]

xs = []
ys = []
for i in range(config["cross_validation_k"]):
    xs.append(eps_per_fold_cum[i]+1)
    ys.append(training_accuracy[eps_per_fold_cum[0]-1])


x1 = list(range(1, len(training_accuracy) + 1))
x2 = list(range(1, len(validation_accuracy) + 1))

fig = plt.figure(figsize=(15, 6))

# Create the plot
plt.plot(x1, training_accuracy, label='Training Accuracy')
plt.plot(x2, validation_accuracy, label='Validation Accuracy')



# Add points with labels
for i in range(config["cross_validation_k"]-1):
    plt.axvline(x=xs[i], color='red', linestyle='--')

# Add annotations to the lines
for i in range(config["cross_validation_k"]-1):
    plt.annotate('Fold '+str(i+2), xy=(xs[i], 0.2), xytext=(xs[i]+1, 0.15),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Add a legend
plt.legend()

# Enable the grid
# plt.grid(True)

## Log Train - Val Accuracy Curve
wandb.log({'Train - Val Accuracy': wandb.Image(plt)})


## Training/Validation Loss
x1 = list(range(1, len(training_loss) + 1))
x2 = list(range(1, len(validation_loss) + 1))

fig = plt.figure(figsize=(15, 6))

# Create the plot
plt.plot(x1, training_loss, label='Training Loss')
plt.plot(x2, validation_loss, label='Validation Loss')


# Add points with labels
for i in range(config["cross_validation_k"]-1):
    plt.axvline(x=xs[i], color='red', linestyle='--')

# Add annotations to the lines
for i in range(config["cross_validation_k"]-1):
    plt.annotate('Fold '+str(i+2), xy=(xs[i], 0), xytext=(xs[i]+1, 0),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Add a legend
plt.legend()

# Enable the grid
# plt.grid(True)

## Log Train - Val Accuracy Curve
wandb.log({'Train - Val Loss': wandb.Image(plt)})


## Log Unified Test Set and its predictions
prediction_names = [class_name_label[num] for num in predictions]
actual_names = [class_name_label[num] for num in y_test.numpy()]

pred_is_actual = [x == y for x, y in zip(prediction_names, actual_names)]


### Log Correct and Wrong Predictions as Images
# assuming you have lists of images, predictions, and actual labels
for i, (img, pred, label) in enumerate(zip(x_test, prediction_names, actual_names)):
    wandb.log({'All Predictions - Unified Test Set': wandb.Image(img, caption='predicted: {}, actual: {}'.format(pred, label))})
    
    if(pred_is_actual[i]):
        wandb.log({'Correct Predictions - Unified Test Set': wandb.Image(img, caption='predicted: {}, actual: {}'.format(pred, label))})
    else:
        wandb.log({'Wrong Predictions - Unified Test Set': wandb.Image(img, caption='predicted: {}, actual: {}'.format(pred, label))})


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


# create a wandb.Table object from the dataframe
table = wandb.Table(dataframe=class_accuracy)
# log the table to wandb
wandb.log({'Class Accuracies': table})


wandb.finish()