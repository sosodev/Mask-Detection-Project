import matplotlib
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import utils.flask_helpers as fh

from flask import Flask, request, Response
from mrcnn import visualize
from utils.model import InferenceConfig
from io import BytesIO
from keras import backend as K

inference_config = InferenceConfig()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return 'Hello, world!'

@app.route('/visualize', methods=['POST'])
def visualize_image():
    K.clear_session()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='.')
    model.load_weights('mask_rcnn_masked_faces.h5', by_name=True)

    image = fh.image_from_request(request)
    image = fh.image_to_array(image)

    results = model.detect([image], verbose=1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['background', 'unmasked', 'masked'], r['scores'])

    buf = BytesIO()
    plt.savefig(buf, format='jpg')

    response = Response()
    response.set_data(buf.getvalue())
    response.headers['Content-Type'] = 'image/jpeg'
    return response
