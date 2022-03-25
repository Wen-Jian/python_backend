import tensorflowjs as tfjs
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'services/'))
from custom_models.P_COVNET import InpaintingModel

model = InpaintingModel().prepare_model()
# tfjs.converters.save_keras_model(model, 'js_models/p_conv')
model.save('keras_models/p_conv.h5', save_format='h5',)
# tf.saved_model.save(model, 'keras_models/whole_model')
# tensorflowjs_converter --input_format=keras keras_models/p_conv.h5 js_models/p_conv
# tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve  keras_models/whole_model/saved_model js_models/web_model