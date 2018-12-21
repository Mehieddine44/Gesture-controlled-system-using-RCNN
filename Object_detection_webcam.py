import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import label_map_util
import visualization_utils as vis_util
import pyautogui as pp


# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = 'frozen_inference_graph.pb'

# Path to label map file
PATH_TO_LABELS = 'labelmap.pbtxt'

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `1`, we know that this corresponds to `finger`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
w = video.get(cv2.CAP_PROP_FRAME_WIDTH);
h = video.get(cv2.CAP_PROP_FRAME_HEIGHT);
out = cv2.VideoWriter('output.avi',fourcc, 5.0, (int(w),int(h)))

#ret = video.set(3,1280)
#ret = video.set(4,720)
###
my_past = 0
mx_past = 0
while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    # resize
    #frame = cv2.resize(frame, (150,150))
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.95)
    
     ###########
    '''
    my_now=boxes[0][0][0]
    mx_now=boxes[0][0][1]
    move_y =(my_now-my_past)*1500
    move_x = (mx_past-mx_now)*1500
    print(np.squeeze(boxes))
    print(move_x)
    if (scores[0][0]>0.85):
     pp.moveRel(move_x, move_y)
    ''' 
     ###########

    # All the results have been drawn on the frame, so it's time to display it.
    #cv2.resize(frame, (1000,700))
    out.write(frame)
    cv2.imshow('Object detector', frame)
    #######
    my_past=boxes[0][0][0]
    mx_past=boxes[0][0][1]
    #######
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
out.release()
cv2.destroyAllWindows()

