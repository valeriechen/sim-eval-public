import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
# from PIL import Image
from sklearn.utils import shuffle
import cv2

from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import regularizers

import PIL.Image
import keras 
import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis


def postprocess(X, color_conversion, channels_first):
    X = X.copy()
    X = iutils.postprocess_images(
        X, color_coding=color_conversion, channels_first=channels_first)
    return X


def image(X):
    X = X.copy()
    return ivis.project(X, absmax=255.0, input_is_positive_only=True)


def bk_proj(X):
    X = ivis.clip_quantile(X, 1)
    return ivis.project(X)


def heatmap(X):
    #X = ivis.gamma(X, minamp=0, gamma=0.95)
    return ivis.heatmap(X)


def graymap(X):
    return ivis.graymap(np.abs(X), input_is_positive_only=True)

def plot_image_grid(grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    file_name=None,
                    figsize=None,
                    dpi=224):
    n_rows = len(grid)
    n_cols = len(grid[0])
    if figsize is None:
        figsize = (n_cols, n_rows+1)

    plt.clf()
    plt.rc("font", family="sans-serif")

    plt.figure(figsize=figsize)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows+1, n_cols], loc=[r+1, c])
            # No border around subplots
            for spine in ax.spines.values():
                spine.set_visible(False)
            # TODO controlled color mapping wrt all grid entries,
            # or individually. make input param
            if grid[r][c] is not None:
                print(r,c)
                ax.imshow(grid[r][c]) #, interpolation='none'
            else:
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # column labels
            if not r:
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=22.5,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

            # row labels
            if not c:
                if row_labels_left != []:
                    txt_left = [l+'\n' for l in row_labels_left[r]]
                    ax.set_ylabel(
                        ''.join(txt_left),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='right',
                    )

            if c == n_cols-1:
                if row_labels_right != []:
                    txt_right = [l+'\n' for l in row_labels_right[r]]
                    ax2 = ax.twinx()
                    # No border around subplots
                    for spine in ax2.spines.values():
                        spine.set_visible(False)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(
                        ''.join(txt_right),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='left'
                    )

    if file_name is None:
        plt.show()
    else:
        print('Saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi, bbox_inches='tight')
        plt.show()

def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    # temp = ret/255.
    # plt.imshow(temp) 
    # plt.savefig("temp.png")
    return ret


# img_height,img_width = 224,224 
# num_classes = 10
# base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape= (img_height,img_width,3), )

# for layer in base_model.layers[:]:
#    layer.trainable = False

# x = base_model.output
# x = Flatten()(x)
# x = Dropout(0.5)(x)
# x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
# x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation= 'softmax')(x)
# model = Model(inputs = base_model.input, outputs = predictions)
# model.load_weights('Dogsfinetune.h5')
# model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

# datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# test_it = datagen.flow_from_directory('../dog_dataset/test/', target_size =(224,224), class_mode='categorical')

input_range = [0, 1]
noise_scale = (input_range[1]-input_range[0]) * 0.1
ri = input_range[0]  # reference input

# images = [load_image(os.path.join("../dog_dataset", "test", "Beagle", f), 224) for f in os.listdir(os.path.join("../dog_dataset", "test", "Beagle"))]

# Configure analysis methods and properties
# methods = [
#     # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

#     # Show input
#     ("input",                 {},                       image,      "Input"),

#     # Function
#     ("gradient",              {"postprocess": "abs"},   graymap,        "Gradient"),
#     ("smoothgrad",            {"noise_scale": noise_scale,
#                                "postprocess": "square"},graymap,        "SmoothGrad"),

#     # Signal
#     ("deconvnet",             {},                       bk_proj,        "Deconvnet"),
#     ("guided_backprop",       {},                       bk_proj,        "Guided Backprop",),
#     # Interaction
#     ("deep_taylor.bounded",   {"low": input_range[0],
#                                "high": input_range[1]}, heatmap,        "DeepTaylor"),
#     ("input_t_gradient",      {},                       heatmap,        "Input * Gradient"),
#     ("integrated_gradients",  {"reference_inputs": ri}, heatmap,        "Integrated Gradients"),
#     #("deep_lift.wrapper",     {"reference_inputs": ri}, heatmap,        "DeepLIFT Wrapper - Rescale"),
#     #("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"},heatmap,        "DeepLIFT Wrapper - RevealCancel"),
#     ("lrp.z",                 {},                       heatmap,        "LRP-Z"),
#     ("lrp.epsilon",           {"epsilon": 1},           heatmap,        "LRP-Epsilon"),
# ]

methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

    # Show input
    ("input",                 {},                       image,      "Input"),

    # Function
    ("gradient",              {"postprocess": "abs"},   graymap,        "Gradient"),
    ("smoothgrad",            {"noise_scale": noise_scale,
                               "postprocess": "square"},graymap,        "SmoothGrad"),
    ("integrated_gradients",  {"reference_inputs": ri}, heatmap,        "Integrated Gradients"),
]


method_gradient = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

    # Function
    ("gradient",              {"postprocess": "abs"},   graymap,        "Gradient"),
]

methodsmoothgrad = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE
("smoothgrad",            {"noise_scale": noise_scale,
                               "postprocess": "square"},graymap,        "SmoothGrad"),
]

method_IG = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE
("integrated_gradients",  {"reference_inputs": ri}, heatmap,        "Integrated Gradients"),
]


def get_saliency_map(exp_type, images, model, dataset):

    if dataset == 'dog':
        all_input = np.zeros((len(images), 224, 224, 3))
        for i in range(len(images)):
            all_input[i] = images[i]   
 
    if dataset == 'mnist':
        all_input = images 

    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

    if exp_type == 'gradient':
        methods= method_gradient
    elif exp_type == 'smoothgrad':
        methods= method_smoothgrad
    elif exp_type == 'IG':
        methods= method_IG

    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                model_wo_softmax, # model without softmax output
                                                **method[1])      # optional analysis parameters

        # Some analyzers require training.
        #analyzer.fit(test_it, batch_size=256, verbose=1)
        analyzers.append(analyzer)

    #x = image
    #x = x[None, :, :, :]

    x_pp = preprocess_input(np.copy(all_input))
    a = analyzer.analyze(x_pp)

    a_list = []

    for image in a:
        a_list.append(methods[0][2](image))
    return a_list

    #a = methods[0][2](a) # post processing!! 
    #return a[0]






# analyzers = []
# for method in methods:
#     analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
#                                             model_wo_softmax, # model without softmax output
#                                             **method[1])      # optional analysis parameters

#     # Some analyzers require training.
#     #analyzer.fit(test_it, batch_size=256, verbose=1)
#     analyzers.append(analyzer)

# num_images = 3

# analysis = np.zeros([num_images, len(analyzers)]+[224,224]+[3])
# text = []

# channels_first = keras.backend.image_data_format() == "channels_first"


# for i in range(num_images):
#     # Add batch axis.
#     x = images[i]
#     x = x[None, :, :, :]

#     # Predict final activations, probabilites, and label.
#     # presm = model_wo_softmax.predict_on_batch(x_pp)[0]
#     # prob = model.predict_on_batch(x_pp)[0]
#     # y_hat = prob.argmax()
    
#     # # Save prediction info:
#     # text.append(("%s" % label_to_class_name[y],    # ground truth label
#     #              "%.2f" % presm.max(),             # pre-softmax logits
#     #              "%.2f" % prob.max(),              # probabilistic softmax output  
#     #              "%s" % label_to_class_name[y_hat] # predicted label
#     #             ))

#     for aidx, analyzer in enumerate(analyzers):
#         #print(aidx)
#         if methods[aidx][0] == "input":
#             # Do not analyze, but keep not preprocessed input.
#             a = x/255.
#             #print(a)
#         elif analyzer:
#             # Analyze.
            
#             x_pp = preprocess_input(np.copy(x))
#             a = analyzer.analyze(x_pp)

#             # Apply common postprocessing, e.g., re-ordering the channels for plotting.
#             #a = postprocess(a, None, channels_first)
#             # Apply analysis postprocessing, e.g., creating a heatmap.
#             a = methods[aidx][2](a)
#             print(a.shape)
#         else:
#             a = np.zeros_like(image)
#         # Store the analysis.
#         print(a[0].shape)
#         analysis[i, aidx] = a[0]

# grid = [[analysis[i, j] for j in range(analysis.shape[1])]
#         for i in range(analysis.shape[0])]  
# # Prepare the labels
# # label, presm, prob, pred = zip(*text)
# label = presm = prob = pred = [0. for i in range(num_images)]
# row_labels_left = [('label: {}'.format(label[i]),'pred: {}'.format(pred[i])) for i in range(len(label))]
# row_labels_right = [('logit: {}'.format(presm[i]),'prob: {}'.format(prob[i])) for i in range(len(label))]
# col_labels = [''.join(method[3]) for method in methods]

# # Plot the analysis.
# plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
#                        file_name="beagles.png")
