import time
import cv2
from datetime import datetime, timedelta

import inference
import utils
import config
import notifications as notifs

def video_inference():
    utils.check_dir_integrity()
    object_det_model = inference.InferenceModel(config.MODEL_PATH)

    try:
        cap = cv2.VideoCapture(config.VIDEO_PATH)
        while True:        
            _, image_np = cap.read()

            infer_results = object_det_model.run_inference(image_np)
            drawn_on_image, max_bb_conf = inference.detection_output_results(image_np, 
                                                                            infer_results, 
                                                                            config.DETECTION_CONF_THRESHOLD)
            
            # if bbox conf > notification threshold:
            # mail function is not on cooldown -> send mail of detection
            # log detection data onto logging.csv
            if max_bb_conf > config.NOTIF_THRESHOLD:

                curr_attempt_send = datetime.now()
                if 'next_allowed_send' not in locals():
                    next_allowed_send = datetime.now() - timedelta(seconds=2)
                
                if curr_attempt_send > next_allowed_send:
                    next_allowed_send = datetime.now() + timedelta(hours=config.NOTIF_HR_COOLDOWN)
                    body = notifs.generate_body()
                    
                    export_path = utils.export_image(drawn_on_image)
                    notifs.send_mail(export_path, body)
                    utils.logging(export_path, max_bb_conf)

    except KeyboardInterrupt:
        pass                

def image_inference():

    utils.check_dir_integrity()
    object_det_model = inference.InferenceModel(config.MODEL_PATH)

    try:
        while True:
            image_name, image_path = utils.rand_image(config.TEST_IMAGES_DIR)
            print(f'Running inference: {image_name}')

            image_np = inference.load_image_into_numpy_array(image_path)
            infer_results = object_det_model.run_inference(image_np)
            drawn_on_image, max_bb_conf = inference.detection_output_results(image_np, 
                                                                             infer_results, 
                                                                             config.DETECTION_CONF_THRESHOLD)
            
            # if bbox conf > notification threshold:
            # mail function is not on cooldown -> send mail of detection
            # log detection data onto logging.csv
            if max_bb_conf > config.NOTIF_THRESHOLD:

                curr_attempt_send = datetime.now()
                if 'next_allowed_send' not in locals():
                    next_allowed_send = datetime.now() - timedelta(seconds=2)
                
                if curr_attempt_send > next_allowed_send:
                    next_allowed_send = datetime.now() + timedelta(hours=config.NOTIF_HR_COOLDOWN)
                    body = notifs.generate_body()

                    export_path = utils.export_image(drawn_on_image)
                    notifs.send_mail(export_path, body)
                    utils.logging(export_path, max_bb_conf)

    except KeyboardInterrupt:
        pass

def inference_type(param):
    """Defines the inference type to run the program as. 
    'image' sets the program to run inference tests on images within a directory specified in the config file.
    'video' sets the program to run inference on a video or attached camera. 

    file path specified can be adjusted in the config file. 
    for video inference for an attached camera, specify '0' as video_path

    Args:
        param: 'image' or 'video'
    """
    if param == 'image': inference.image_inference()
    elif param == 'video': inference.video_inference()
    else: print('Invalid Input: image/video')

def main():
    inference_category = 'image'
    inference_type(inference_category)

if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Program has been running for {"%.2f"%(elapsed_time/3600)}hrs')