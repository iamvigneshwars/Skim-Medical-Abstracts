from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    
    return render_template('main.html')


if __name__ == "__main__":
    app.run(debug=True)