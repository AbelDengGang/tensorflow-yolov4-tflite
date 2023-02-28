# coding=utf-8
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'   # 减少日志
import tensorflow as tf
# 必须要有这句话，否则会报错： tensorflow.python.framework.errors_impl.NotFoundError: No algorithm worked! [Op:Conv2D]
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.python.saved_model import tag_constants
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes,test_tflite,test_tf
import core.utils as utils
from core.config import cfg
from core.reader import parse_net
flags.DEFINE_string('cfg', '', 'path to cfg file, if none, only support offical yolo-v3/v4 weights')
flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('test', False, 'test model after convert')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
  bbox_tensors = []
  prob_tensors = []
  # 没有 指定cfg文件，则只能加载官方的yolov3 v4网络模型
  if FLAGS.cfg == '': 
    if FLAGS.tiny:
      for i, fm in enumerate(feature_maps):
        if i == 0:
          output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        else:
          output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])
    else:
      for i, fm in enumerate(feature_maps):
        if i == 0:
          output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        elif i == 1:
          output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        else:
          output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if FLAGS.framework == 'tflite':
      pred = (pred_bbox, pred_prob)
    else:
      boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
      pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
    model.summary()
  else:
    # 从cfg文件读取模型
    global_cfg,input_layer, yolo_layers = parse_net(cfg=FLAGS.cfg,weights=FLAGS.weights)
    model_input_size = int(global_cfg['width'])
    total_layers = len(yolo_layers)
    first_layer_param = yolo_layers[0][1]
    NUM_CLASS = int(first_layer_param['classes'])
    anchors_org = np.array([int(anchor) for anchor in first_layer_param['anchors']],dtype='int').reshape(-1,2)
    # print("anchors_org", anchors_org)
    # 遍历网络参数，准备STRIDES列表和ANCHOR列表
    strides = []
    anchors_decoder = []
    for net,param in yolo_layers:
      size = int(net.shape[-2])
      # strides 是每一层yolo的缩放倍数
      strides.append (model_input_size // size)

      # 每一层YOLO使用的ANCHOR不是按照顺序使用的，所以要重新按照yolo层来组织ancho列表
      mask=[int(mask_str) for mask_str in param['mask']]
      anchor_layer = [anchors_org[mask_idx] for mask_idx in mask]
      anchors_decoder.append(anchor_layer)
      # print("mask,", mask)
    anchors_decoder = np.array(anchors_decoder)
    # print("anchors_decoder", anchors_decoder)
    # print("strides",strides)

    bbox_tensors = []
    prob_tensors = []
  
    for i ,yolo_layer in enumerate(yolo_layers):
      net,param = yolo_layer
      size = int(net.shape[-2])
      output_tensors = decode(net, size, NUM_CLASS, strides, anchors_decoder, i, XYSCALE, 'tf')
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    model.summary()

  if FLAGS.test:
    import cv2
    from PIL import Image
    input_size = FLAGS.input_size
    image_path = FLAGS.image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    if FLAGS.framework == 'tf':
      # test_tf 中， 
      # 如果 infer 是 <class 'tensorflow.python.saved_model.load._WrapperFunction'> ，返回dict
      # 如果 infer 是 model 返回 tensor
      # 原来的代码只对dic数据进行解析，因此需要通过先保存再读取的方法来把model转成_WrapperFunction
      model.save(FLAGS.output)
      saved_model_loaded = tf.saved_model.load(FLAGS.output, tags=[tag_constants.SERVING])
      infer = saved_model_loaded.signatures['serving_default']
      pred_bbox = test_tf(image_data,input_size,infer,FLAGS)

    image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    # cv2.imwrite(FLAGS.output, image)
  else:
    model.save(FLAGS.output)



def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
