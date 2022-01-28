#!/usr/bin/env python3

"""Web application startup file.

References
----------
app
    A package that contains all the components of a web application,
    in the file __init__ of which an instance of the WSGI application
    is created.
"""

from app import app

if __name__ == '__main__':
    app.run()
