from distutils.log import debug
# from flask_script import Manager, Server
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'app/'))
from app import main
from app.models.user import User
from waitress import serve

# manager = Manager(main)
# manager.add_command('runserver', serve(main, port=5000))

# @manager.shell
# def make_shell_context():
#     return dict(app=main, User=User)

if __name__ == '__main__':
    # manager.run()
    serve(main, port=5000)