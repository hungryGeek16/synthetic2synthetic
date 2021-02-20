import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

di = {}

with tf.Graph().as_default() as graph: # Set default graph as graph
    with tf.Session() as sess:
    # Load the graph in graph_def
        print("load graph")

    # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
        with gfile.FastGFile("frozen_inference_graph.pb",'rb') as f:
     # Set FCN graph to the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

    # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

            tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
            )         
        
        
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')
        
        image = cv2.imread("test1.jpg")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        
        (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

label_map = label_map_util.load_labelmap("labelmap.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=330, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=4,
    min_score_thresh=0.30)

# All the results have been drawn on image. Now display the image.
cv2.imshow("detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
