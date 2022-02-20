from flask import Flask, session
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv('SECRET_KEY')
app.secret_key = "HelloWorld"

from ssebowa import routes
