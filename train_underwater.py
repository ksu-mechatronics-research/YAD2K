"""
This is a sample training script used to test the implementation of the
YOLO localization loss function.
"""
import io
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model

from keras.callbacks import TensorBoard, ModelCheckpoint

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_data(DATA_PATH):
    '''loads the data'''
    data = h5py.File(DATA_PATH, 'r')

    images = data['train/images']
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]


    # Box preprocessing.
    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
    boxes = np.array(data['train/boxes'])
    boxes = [box.reshape((-1, 5)) for box in boxes]
    # Get extents as y_min, x_min, y_max, x_max, class for comparision with
    # model output.
    boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

    # Get box parameters as x_center, y_center, box_width, box_height, class.
    boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
    boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
    boxes_xy = boxes_xy / orig_size
    boxes_wh = boxes_wh / orig_size
    boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

    return images, processed_images, boxes

def get_detector_mask(boxes, anchors):
    # Precompute detectors_mask and matching_true_boxes for training.
    # Detectors mask is 1 for each spatial position in the final conv layer and
    # anchor that should be active for the given boxes and 0 otherwise.
    # Matching true boxes gives the regression targets for the ground truth box
    # that caused a detector to be active or 0 otherwise.
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return detectors_mask, matching_true_boxes

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True, count=13):
    '''returns the model'''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names), count)
    # TODO: Fix this bug
    # calling yolo_body twice in one session causes error:
    # File "/home/sexy/YAD2K/yad2k/models/keras_yolo.py", line 52, in yolo_body
    #   conv13 = darknet.get_layer('leaky_re_lu_13').output
    # yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [image_input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model



def _main():
    DATA_PATH = os.path.expanduser(os.path.join('..', 'DATA', 'underwater.hdf5'))
    classes_path = os.path.expanduser(os.path.join('model_data','underwater_classes.txt'))

    class_names = get_classes(classes_path)
    images, image_data, boxes = get_data(DATA_PATH)

    anchors = YOLO_ANCHORS

    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    model_body, model = create_model(anchors, class_names)

    # #################################################
    # Train the model
    # #################################################
    image_data = np.array(image_data)
    boxes = np.array(boxes)
    detectors_mask = np.asarray(detectors_mask)
    matching_true_boxes = np.array(matching_true_boxes)

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging = TensorBoard()
    checkpoint = ModelCheckpoint("training_intermediate.h5", monitor='loss',
                                 save_weights_only=True, save_best_only=True)

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              batch_size=32,
              epochs=5,
              callbacks=[logging, checkpoint])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False, count=35)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

<<<<<<< HEAD
<<<<<<< HEAD
    # model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
    #           np.zeros(len(image_data)),
    #           batch_size=8,
    #           epochs=num_steps,
    #           callbacks=[logging, checkpoint])
    # model.save_weights('overfit_weights.h5')
    model.load_weights('overfit_weights.h5')

=======
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              batch_size=8,
              epochs=1000,
              callbacks=[logging, checkpoint])

    model.save_weights('trained.h5')

    # model.load_weights('training_intermediate.h5')
>>>>>>> 33bc1a651a22b7d1f1924d1f7075d748a01a9281

    image_data = [np.expand_dims(image, axis=0) for image in image_data]
    boxes = [np.expand_dims(box, axis=0) for box in boxes]
    detectors_mask = [np.expand_dims(mask, axis=0) for mask in detectors_mask]
    matching_true_boxes = [np.expand_dims(box, axis=0) for box in matching_true_boxes]

    image_data = np.array(image_data)
    boxes = np.array(boxes)
    detectors_mask = np.asarray(detectors_mask)
    matching_true_boxes = np.array(matching_true_boxes)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.05, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    out_path = "underwater_images"

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(images)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [images[i].size[1], images[i].size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        if len(out_boxes) > 0:
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()


if __name__ == '__main__':
    # create_model(YOLO_ANCHORS, get_classes(os.path.expanduser(os.path.join('model_data','underwater_classes.txt'))))
    _main()
