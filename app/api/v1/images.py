from bdb import set_trace
from flask import Blueprint, jsonify
from flask import request
import base64
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'services/'))
from custom_models.P_COVNET import InpaintingModelV2
from image_processor import ImageProcessor
import cv2
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

imagesRoute = Blueprint('imagesRoute', __name__)

@imagesRoute.route('/inpaint_image', methods=['POST'])
def inpaint_image():
    params = request.json
    img_data = params.get('img_data')
    line_weight = params.get('line_weight')
    paths = [json.loads(params.get('paths'))]
    binary = base64.b64decode(img_data)
    image = np.asarray(bytearray(binary), dtype="uint8")
    image_np = cv2.imdecode(image, cv2.IMREAD_COLOR)

    img_shape = image_np.shape
    padding_y_n = 32 - img_shape[0] % 32
    padding_x_n = 32 - img_shape[1] % 32
    padded_imgs = np.array([np.stack([np.pad(image_np[:,:,c], pad_width=((0,padding_y_n), (0,padding_x_n)), mode='constant', constant_values=255) for c in range(3)], axis=2)])

    # model_v1 = InpaintingModel().prepare_model()
    # model_v1.compile(optimizer='adam', loss='mean_absolute_error')
    # masked_xs, ys = ImageProcessor.mask_data(padded_imgs, 1, 5)
    # inpainted_image_v1 = model_v1.predict(np.expand_dims(masked_xs[0], axis=0))[0]
    # inpainted_image = (inpainted_image_v1 * 255).astype(np.uint8)
    # retval, buffer = cv2.imencode('.jpg', inpainted_image[0:img_shape[0], 0:img_shape[1], :])

    padded_img_shape = padded_imgs[0].shape
    model_v2 = InpaintingModelV2().prepare_model(padded_img_shape)
    model_v2.compile(optimizer='adam', loss='mean_absolute_error')
    [masked_xs_v2, masks], ys = ImageProcessor.mask_data_v2(padded_imgs, 1, paths, line_weight)
    sample_idx = 0
    inpainted_image_v2 = model_v2.predict([masked_xs_v2[sample_idx].reshape((1,)+masked_xs_v2[sample_idx].shape), masks[sample_idx].reshape((1,)+masks[sample_idx].shape)])[0]
    inpainted_image_v2 = (inpainted_image_v2 * 255).astype(np.uint8)
    retval, buffer = cv2.imencode('.jpg', inpainted_image_v2[0:img_shape[0], 0:img_shape[1], :])
    
    new_image_string = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        'result': 'success',
        'img_data': new_image_string
    }), 200

# def imgs():
#     return send_file(, mimetype='image/jpeg')