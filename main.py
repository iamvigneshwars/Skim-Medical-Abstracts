from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    return "This is a main page"


if __name__ == "__main__":
    app.run(debug=True)