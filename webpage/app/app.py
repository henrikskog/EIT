from flask import Flask, render_template, jsonify
import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PARENT_DIR)

#Bruk denne importen når man kjører ilage med camera
#Må også endre i jsonify til webpage.shared_data.cnt_xx. Linje 25 og 26
#import webpage.shared_data
import shared_data

app = Flask(__name__)

cnt_up = 0
cnt_down = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/counter')
def counter():
    return jsonify({
        'upCount': shared_data.cnt_up,
        'downCount': shared_data.cnt_down
    })

def run_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    print("Starting Flask app...")
    run_flask_app()
