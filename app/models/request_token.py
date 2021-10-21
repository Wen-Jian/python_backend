from db import db

class RequestToken(db.Model):
    __table_args__ = {"useexisting": True}
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(16))
    expired_datetime = db.Column(db.DateTime)

    def __repr__(self):
        return '<RequestToken {}, {}>'.format(self.id, self.token) 
    
    def __init__(self, token, expired_datetime):
        self.token = token
        self.expired_datetime = expired_datetime