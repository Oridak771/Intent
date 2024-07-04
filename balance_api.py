from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/balance', methods=['GET'])
def get_balance():
    response = {
        'account_balance': '$500'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
