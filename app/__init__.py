from flask import Flask, make_response
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'app/'))
sys.path.append(os.path.join(os.getcwd(), 'services/'))
from db import db
from flask_migrate import Migrate
from flask_cors import CORS

from api.v1.login import loginRoute
from api.v1.images import imagesRoute

main = Flask(__name__)
main.register_blueprint(loginRoute)
main.register_blueprint(imagesRoute)
main.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

db.init_app(main)
migrate = Migrate(main, db)

from app.models.user import User
from app.models.request_token import RequestToken
CORS(main)

@main.after_request # blueprint can also be app~~
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response


if __name__ == '__main__':
    main.run(port=5000, debug=True)
    