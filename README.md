from flask import Flask, request, jsonify
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy, os, io
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from autocorrect import Speller
from datetime import datetime
from transformers import pipeline
from translate import Translator
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from googletrans import Translator
from google.cloud import speech

#_______________________________________________________________________________________________________________________#

translator = Translator()

#transformers that can be used to classify text sequences
zero_shot_classifier = pipeline('zero-shot-classification', model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
pd.set_option('display.max_colwidth', None)
data1 = pd.read_csv('final dataset.csv')
spell_checker = Speller(lang='en')
class_names = ["positive :)", "neutral :|", "negative :("]

#____________________________________________FUNCTIONS__________________________________________________________________#

def detect_language(user_input):
    det = translator.detect(user_input)
    if det.lang!='en':
        trans = translator.translate(user_input,'en')
        print("\nTranslation:",trans.text)
        return trans.text
    else:
        return user_input

def remove_stopwords(tags):
    words = word_tokenize(tags)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = " ".join(filtered_words)
    return filtered_text

def porterStemmer(text):
    words = word_tokenize(text)
    stemmed_words = [porter.stem(word) for word in words]
    stemmed_sentence = ' '.join(stemmed_words)
    return stemmed_sentence

def correct_spelling(word):
    return spell_checker(word)

def correct_spellings_in_text(text):
    words = nltk.word_tokenize(text)
    corrected_words = [correct_spelling(word) for word in words]
    corrected_text = " ".join(corrected_words)
    return corrected_text

def preprocess_input(userInput):
    corrected_text = correct_spellings_in_text(userInput)
    words = nltk.word_tokenize(corrected_text.lower())
    sentence = " ".join(words)
    sentence = remove_stopwords(sentence)
#     sentence = porterStemmer(sentence)
    keywords = nltk.word_tokenize(sentence.lower())
    return keywords, sentence

def calculate_score(about, keywords):
    score = 0
    for keyword in keywords:
        if keyword in about.lower():
            score += 1
    return score

def zero_shot_classifier_sent(userInput):
    zsc_output = zero_shot_classifier(userInput, class_names)
    zsc_labels = zsc_output['labels']
    zsc_scores = zsc_output['scores']
    return zsc_labels, zsc_scores

def recommendArticle(userInput, tfidf_scores, output_csv):
    zsc_labels, zsc_scores = zero_shot_classifier_sent(userInput)
    label_score_pairs = zip(zsc_labels, zsc_scores)
    max_label, max_score = max(label_score_pairs, key=lambda pair: pair[1])
    userInput = detect_language(userInput) #change to english
    keywords, sentence = preprocess_input(userInput)
    data1['score'] = data1['description'].apply(lambda x: calculate_score(x, keywords))

    # Sort articles based on score
    recommended_articles = data1.sort_values(by='score', ascending=False)

    print("\nSentiment Score")
    for i in range(0,3):
        print(f"{zsc_labels[i]}: {zsc_scores[i]}")
    
    print("\nEmotion:", max_label)
    print("Score:", max_score)
    print("\nKeywords:", keywords)
    
    print("\n*****************\nRecommended Articles:")
    for index, row in recommended_articles.head(10).iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Keywords: {row['keywords']}")
        print(f"Class: {row['class']}")
        print(f"URL: {row['url']}")
    
    # Prepare data to append to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_data = {
        'Timestamp': timestamp,
        'User Input': userInput,
        'Emotion': max_label,
        'Sentiment Score': max_score,
        'Keywords': ", ".join(keywords)}

    # Append output data to CSV
    output_df = pd.DataFrame(output_data, index=[0])
    output_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

def convert_audio_to_text(recognizer, source, duration):
    print("Listening for audio...")
    audio_data = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.WaitTimeoutError:
        print("Listening timed out. No speech detected.")
        return ""
    except sr.UnknownValueError:
        print("Oops, it seems we're having trouble understanding the audio. Let's try again with clearer sound.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""
    
def audio_to_speech(audioFile):
    with io.open(audio_file,'rb') as f:
        content = f.read()
        audio = speech.RecognitionAudio(content=content)
        

def extract_keywords_tfidf(article_descriptions):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(article_descriptions)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    article_tfidf_scores = tfidf_matrix[0].toarray().flatten()
    keyword_scores = dict(zip(feature_names, article_tfidf_scores))
    return keyword_scores

#__________________________________________________MAIN_________________________________________________________________#

def main(inputs):
    output_csv = "Output2.csv"  # Specify the output CSV file
    print("Choose input method:\n1. Text\n2. Voice\n3. Audio File")
    while True:
        choice = input("\nEnter your choice (1 or 2 or 3): ")

        if choice == '1':
            user_input1 = input("Enter your message: ")
            user_input1 = detect_language(user_input1)
            inputs.append(user_input1)
            user_input = ' '.join(inputs)
            print(user_input)
            print("\nProcessing....")
            tfidf_scores = extract_keywords_tfidf(data1['description'])
            recommendArticle(user_input, tfidf_scores, output_csv)
            break
            
        elif choice == '2':
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
                text1 = convert_audio_to_text(recognizer, source, 15)

                if text1:
                    text = detect_language(text1)
                    inputs.append(text1)
                    text = ' '.join(inputs)
                    print(text)
                    print("\nProcessing....")
                    tfidf_scores = extract_keywords_tfidf(data1['description'])
                    recommendArticle(text, tfidf_scores, output_csv)
                    break
                else:
                    print("Oops, it seems we're having trouble understanding the audio. Let's try again with clearer sound.")
                
        elif choice == '3':
            filename = input("Enter the path to the audio file: ")
            recognizer = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
                text1 = convert_audio_to_text(recognizer, source, 1000)

                if text1:
                    text = detect_language(text1)
                    inputs.append(text1)
                    text = ' '.join(inputs)
                    print(text) 
                    print("\nProcessing....")
                    tfidf_scores = extract_keywords_tfidf(data1['description'])
                    recommendArticle(text, tfidf_scores, output_csv)
                    break
                else:
                    print("Oops, it seems we're having trouble finding the file. Let's try again with the correct path.")
        else:
            print("Invalid choice. Please enter 1 or 2 or 3.")

if __name__ == "__main__":
    inputs = []
    main(inputs)
    while True:
        user_command = input("\n1. Not what you're looking for? Continue adding detail?\n2. Ask other topic\n3. Exit.\n\nEnter your choice: ")
        if user_command.lower() == '1':
            print("\n")
            main(inputs)
        elif user_command.lower() == '2':
            inputs = []
            print("\n")
            main(inputs)
        elif user_command.lower() == '3':
            print("\n*********Thank you for using our service.*********")
            break
        else:
            print("\nPlease type in the correct command.")
    
# Initialize Flask app
app = Flask(__name__)

# Define API endpoints
@app.route('/text', methods=['POST'])
def text_input():
    data = request.get_json()
    user_input = data['text']
    emotion, sentiment_score = analyze_sentiment_vader(user_input)
    tfidf_scores = extract_keywords_tfidf(data1['description'])
    recommendations = recommendArticle(user_input, emotion, sentiment_score, tfidf_scores)
    return jsonify(recommendations)

@app.route('/audio', methods=['POST'])
def audio_input():
    recognizer = sr.Recognizer()
    with sr.AudioFile(request.files['audio']) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_text = recognizer.recognize_google(source)
        emotion, sentiment_score = analyze_sentiment_vader(audio_text)
        tfidf_scores = extract_keywords_tfidf(data1['description'])
        recommendations = recommendArticle(audio_text, emotion, sentiment_score, tfidf_scores)
        return jsonify(recommendations)

@app.route('/classify-text', methods=['POST'])
def classify_text():
    data = request.get_json()
    user_input = data['text']
    labels, scores = zero_shot_classifier_sent(user_input)
    return jsonify({'labels': labels, 'scores': scores})

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    user_input = data['text']
    translated_text = detect_language(user_input)
    return jsonify({'translated_text': translated_text})

# Run the Flask app
if __name__ == '__main__':

    app.run(debug=True)
