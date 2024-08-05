import json
import base64
import io
from PIL import Image

import torch
from detectron2.model_zoo import get_config
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

CONFIG_OPTS = ["MODEL.WEIGHTS", 'model_final.pth']
CONFIDENCE_THRESOLD = 0.5
CLASSES = {0: 'bloc'}

def init_context(context):
    context.logger.info('Init context... 0%')
    
    CONFIG_OPTS.extend(['MODEL.DEVICE', 'cpu'])
    
    ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
    CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
    cfg = get_config(CONFIG_FILE_PATH)
    
    cfg.merge_from_list(CONFIG_OPTS)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESOLD
    cfg.freeze()
    predictor = DefaultPredictor(cfg)    
    
    context.user_data.moodel_handler = predictor
    
    context.logger.info('Init context...100%')
    

def handler(context, event):
    context.logger.info('Run mask_rcnn_R101 model')
    data = event.body()
    buf = io.BytesIO(base64.b64decode(data['image']))
    thresold = float(data.get('thresold', 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format='BGR')
    
    predictions = context.user_data.model_handler(image)
    
    instances = predictions['instances']
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    results = []
    for box, score, label in zip(pred_boxes, scores, pred_classes):
        label = CLASSES[int(label)]
        if score >= thresold:
            results.append({
                'confidence': str(float(score)),
                'label': label,
                'points': box.tolist(),
                'type': 'polygon'
            })
    
    return context.Response(body=json.dumps(results), headers={}, content_type='application/json', status_code=200)