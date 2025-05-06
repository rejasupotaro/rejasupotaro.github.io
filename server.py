from flask import Flask, send_from_directory

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

@app.route('/components/<path:filename>')
def serve_components(filename):
    return send_from_directory('components', filename)

@app.route('/pages/<path:filename>')
def serve_pages(filename):
    return send_from_directory('pages', filename)

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, port=8000) 