import tensorflow as tf
import numpy as np
import cv2
import time

import config

class InferenceModel:
    def __init__(self, model_path):
        print('Loading model.. ')
        start_time = time.time()
        
        self.__load_model(model_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Load Time: {} secs'.format("%.2f" %elapsed_time))

    def run_inference(self, image: np.ndarray):
        if self.detect_fn != None:
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis, ...]
            return self.detect_fn(input_tensor)

    def __load_model(self, model_path):
        self.detect_fn = tf.saved_model.load(model_path)

def load_image_into_numpy_array(path):
    return cv2.imread(path)

def denormalize_image_bbox(bbox, im_height, im_width):
    ymin, xmin, ymax, xmax = bbox
    xmin, xmax, ymin, ymax = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    return [xmin, xmax, ymin, ymax]

def draw_bbox(image_np, bbox, display_text):
    xmin, xmax, ymin, ymax = bbox
    cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.putText(image_np, display_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    line_width = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
    cv2.line(image_np, (xmin, ymin), (xmin + line_width, ymin), (0, 255, 0), thickness=5)
    cv2.line(image_np, (xmin, ymin), (xmin, ymin + line_width), (0, 255, 0), thickness=5)

    cv2.line(image_np, (xmax, ymin), (xmax - line_width, ymin), (0, 255, 0), thickness=5)
    cv2.line(image_np, (xmax, ymin), (xmax, ymin + line_width), (0, 255, 0), thickness=5)

    cv2.line(image_np, (xmin, ymax), (xmin + line_width, ymax), (0, 255, 0), thickness=5)
    cv2.line(image_np, (xmin, ymax), (xmin, ymax - line_width), (0, 255, 0), thickness=5)

    cv2.line(image_np, (xmax, ymax), (xmax - line_width, ymax), (0, 255, 0), thickness=5)
    cv2.line(image_np, (xmax, ymax), (xmax, ymax - line_width), (0, 255, 0), thickness=5)

def detection_output_results(image_np, infer_results, threshold):
    im_height, im_width, _ = image_np.shape
    
    bboxs = infer_results['detection_boxes'][0].numpy()
    class_scores = infer_results['detection_scores'][0].numpy()
    
    bboxs_id = tf.image.non_max_suppression(bboxs, 
                                            class_scores, 
                                            max_output_size=50, 
                                            iou_threshold=threshold, 
                                            score_threshold=threshold)
    
    top_conf, top_bb = 0, []
    if len(bboxs_id) != 0:
        for i in bboxs_id:
            bbox = bboxs[i].tolist()
            bbox = denormalize_image_bbox(bbox, im_height, im_width) # [xmin, xmax, ymin, ymax]
            bbox_confidence = round(100*class_scores[i])
            
            if bbox_confidence > top_conf:
                top_conf = bbox_confidence
                top_bb = bbox

    if top_conf > config.DETECTION_CONF_THRESHOLD:
        display_text = f'Package: {top_conf}%'
        draw_bbox(image_np, top_bb, display_text)

    return image_np, top_conf
