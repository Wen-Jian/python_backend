# from flask import Blueprint, jsonify
# from app.models.user import User
# from app.models.request_token import RequestToken
# from flask import request
# from app import db
# import random
# import string
# from datetime import datetime, timedelta
# from sqlalchemy import desc
# loginRoute = Blueprint('loginRoute', __name__)

# def generateRequestToken():
#     letters = string.ascii_lowercase
#     user_no = ''.join(random.choice(letters) for i in range(8))
#     request_token = ''.join(random.choice(letters) for i in range(16))
#     return request_token, user_no

# @loginRoute.route('/login', methods=['POST'])
# def login():
#     params = request.json
#     email = params.get('account')
#     password = params.get('password')

#     if email == None or password == None:
#         return jsonify({
#             'error': 'accout or password can\'t be blank'
#         }), 400

#     user = User.query.filter_by(email = email).first()

#     if user != None and user.password == password:
#         request_token, _ = generateRequestToken()
#         request_token_record = RequestToken(
#             token=request_token,
#             expired_datetime=datetime.now() + timedelta(days=1)
#         )
#         db.session.add(request_token_record)
#         db.session.commit()
#         user.request_token_id = request_token_record.id
#         db.session.add(user)
#         db.session.commit()
#         return jsonify({
#             'user_name': user.username,
#             'user_no': user.user_no,
#             'request_token': user.request_token.token
#         })
#     elif user == None and email != None and email != '':
#         request_token, user_no = generateRequestToken()
#         request_token_record = RequestToken(
#             token=request_token,
#             expired_datetime=datetime.now() + timedelta(days=1)
#         )
#         user = User(
#             username=params.get('user_no'),
#             user_no=user_no,
#             email=params.get('account'),
#             password=params.get('password'),
#             request_token_id=None
#         )
#         db.session.add(request_token_record)
#         db.session.commit()
#         user.request_token_id = request_token_record.id
#         db.session.add(user)
#         db.session.commit()
#         return jsonify({
#             'user_name': user.username,
#             'user_no': user.user_no,
#             'request_token': user.request_token.token
#         })
#     else:
#         return jsonify({
#             'error': 'account or password is incorrect'
#         }), 400

# # @loginRoute.route('/request_token')
# # def request_token():
# #     params = request.args
# #     request_token = RequestToken.query.filter_by(token = params.get('request_token'))
# #     if request_token.count() > 1 and request_token.order_by(desc(RequestToken.id)).get(first()):
# #         return jsonify({
# #             'user_name': user.username,
# #             'user_no': user.user_no,
# #             'request_token': 'test_token'
# #         })
# #     return jsonify({
# #         'error': 'not found'
# #     }), 404