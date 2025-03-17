import warnings
warnings.simplefilter('ignore')
import os
import pandas, unidecode, json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from flask import Flask, render_template, request

app = Flask(__name__)


class Main:
    def __init__(self):
        # Init
        dataset_path, separator = [os.path.join('model', "dataset.txt"), "   "]
        # Load and Train model
        self.dataset = pandas.read_csv(dataset_path, names=['sentence', 'label'], sep=separator)
        self.vectorizer = None
        self.score = None
        self.model = None
        self.train()
        
        
    def train(self):

        # Separate dataset and expected output
        sentences = self.dataset['sentence'].values
        y = self.dataset['label'].values

        # Split datasets
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

        # Verctorization of training and testing data
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(sentences_train)
        X_train = self.vectorizer.transform(sentences_train)
        X_test  = self.vectorizer.transform(sentences_test)

        print("X_train shape: ", X_train.shape)

        # Init model and fit it
        self.model = XGBClassifier(max_depth=2, n_estimators=30) 
        # max_depth=2 limite de profondeur de l'arbre, 
        # n_estimators=30 nombre d'estimateur permet d'identifier le nombre de critères d'évaluation qu'on s'autorise à utiliser
        self.model.fit(X_train, y_train)

        # Show xboost parameters
        print(self.model)
        
        # make predictions for test data
        y_pred = self.model.predict(X_test)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        self.score = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (self.score * 100.0))
    
    def predict(self, json_text):
        # predictions
        result = self.vectorizer.transform([unidecode.unidecode(json_text)])
        result = self.model.predict(result)

        if str(result[0]) == "0":
            sentiment = "NEGATIVE"

        elif str(result[0]) == "1":
            sentiment = "POSITIVE"

        return json.dumps({"sentiment": sentiment, "text": json_text})
    
main = Main()

@app.route('/')
def index():
    return 'Sentiments Analysis'
    # return ("Index de ma page") renvoie que du texte
    #return render_template('Sentiments_Analysis.html') # renvoie une page html, obligatoir si on doit faire un formulaire : doit faire un fichier html avec l'instruction form


@app.route('/train')
def train():
    main.train()
    return "Model trained"

@app.route('/predict', methods=['POST'])
def predict():
    data=request.json
    return str(main.predict(data['text']))


if __name__ == "__main__":
    # main = Main()
    # print(main.predict("Depuis ce matin votre application ne marche pas, je n'arrive pas à déverrouiller ma voiture."))
    # print(main.predict("j'ai adore la prestation"))

    app.run(host='0.0.0.0', port=80)
    