from distutils.log import debug
from flask_script import Manager, Server
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'app/'))
from app import main
from app.models.user import User

manager = Manager(main)
manager.add_command('runserver', Server())

@manager.shell
def make_shell_context():
    return dict(app=main, User=User)

if __name__ == '__main__':
    manager.run(debug=True)