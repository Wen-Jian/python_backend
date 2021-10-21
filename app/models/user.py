import pdb
# pdb.set_trace()
# from app import db
# import os
# import sys
# sys.path.append(os.path.join(os.getcwd(), 'app/'))
from db import db
from app.models.request_token import RequestToken

class User(db.Model):
    __table_args__ = {"useexisting": True}
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password = db.Column(db.String(128))
    user_no = db.Column(db.String(64))
    request_token_id = db.Column(db.Integer, db.ForeignKey('request_token.id'), index=True, unique=True)
    request_token = db.relationship("RequestToken")

    def __repr__(self):
        return '<User {}>'.format(self.id) 
    
    def __init__(self, username, email, password, user_no, request_token_id):
        self.username = username
        self.email = email
        self.password = password
        self.user_no = user_no
        self.request_token_id = request_token_id