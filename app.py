import os
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data
import heapq

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

nltk.download('punkt')
nltk.download('stopwords')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        num_sentences = int(request.form['num_sentences'])

        if uploaded_file:
            text = uploaded_file.read().decode('utf-8')

            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            stop_words = set(stopwords.words('english'))

            words = [word for word in words if word.lower() not in stop_words]

            word_freq = {}
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

            sentence_scores = {}
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word in word_freq:
                        if sentence not in sentence_scores:
                            sentence_scores[sentence] = word_freq[word]
                        else:
                            sentence_scores[sentence] += word_freq[word]

            summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            summary = ' '.join(summary_sentences)

            return render_template('summary.html', summary=summary)

    return "No file uploaded."

if __name__ == '__main__':
    app.run(debug=True)
