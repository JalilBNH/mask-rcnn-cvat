import json
import base64
import io
from PIL import Image
import numpy as np
import cv2

import torch
from detectron2.model_zoo import get_config
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.config import get_cfg
from detectron2 import model_zoo


WEIGHTS_PATH = "model_final.pth"
CONFIDENCE_THRESHOLD = 0.5
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"


def init_context(context):
    context.logger.info("Init context...  0%")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    context.user_data.model_handler = predictor

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run mask_rcnn model")
    data = event.body
    
    if "image" not in data:
        context.logger.error("No image found in the event payload")
        return context.Response(body=json.dumps({"error": "No image provided"}),
                                content_type='application/json', status_code=404)
    
    try:
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = convert_PIL_to_numpy(Image.open(buf), format="BGR")
        context.logger.info(f"Image received successfully with shape: {image.shape}")
    except Exception as e:
        context.logger.error(f"Failed to decode image: {str(e)}")
    
    try:
        predictions = context.user_data.model_handler(image)
    except Exception as e:
        context.logger.error(f"Error during model inference: {str(e)}")
        
    instances = predictions['instances']
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    pred_masks = instances.pred_masks

    results = []
    
    for box, score, label, mask in zip(pred_boxes, scores, pred_classes, pred_masks):
        label = 'bloc'
        binary_mask = mask.to(torch.uint8).numpy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            contour = contours[0]
            if contour.shape[0] >= 3:  
                
                points = contour.reshape(-1).tolist()  
                results.append({
                    "confidence": str(float(score)),
                    "label": label,
                    "points": points,
                    "type": "polygon",
                })
            else:
                context.logger.warn(f"Contour with less than 3 points found, skipping: {contour}")
        else:
            context.logger.warn("No contours found in the mask, skipping this instance.")
    
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)