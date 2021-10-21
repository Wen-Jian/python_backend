from flask import Blueprint, jsonify, send_file
imagesRoute = Blueprint('imagesRoute', __name__)

@imagesRoute.route('/get_imgs_list')
def get_imgs_list():
    return jsonify({
            'success': True,
            'result': [
                '1', '2'
            ]
        })

# def imgs():
#     return send_file(, mimetype='image/jpeg')