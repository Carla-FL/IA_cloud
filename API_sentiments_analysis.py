from flask import Flask, render_template, request
from main import Main

app = Flask(__name__)
main = Main()

@app.route('/')
def index():
    #return 'Sentiments Analysis'
    # return ("Index de ma page") renvoie que du texte
    return render_template('Sentiments_Analysis.html') # renvoie une page html, obligatoir si on doit faire un formulaire : doit faire un fichier html avec l'instruction form

@app.route('/analysis', methods=['GET'])
def analysis():
    query = request.args.get("query")
    main = Main()
    return f'Analyse de {main.predict(query)}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)