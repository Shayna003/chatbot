import asyncio
import websockets
import spacy
import pickle
from textblob import TextBlob
from gensim.models import KeyedVectors

# Load SpaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Load pre-trained GloVe embeddings
model = KeyedVectors.load_word2vec_format('glove.6B.50d.word2vec.txt', binary=False)

# Load the pipeline, label encoder, and words_by_intent
with open('model_en/pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('model_en/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('model_en/words_by_intent.pkl', 'rb') as f:
    words_by_intent = pickle.load(f)

# Define responses using a dictionary
responses = {
    "greeting": "Hello! I'm smart chatbot Tubby. how can I help you today? :)",
    "ask_name": "My name is Tubby, a chatbot created to assist you. :)",
    "farewell": "Goodbye! Have a great day! ;)",
    "about_ebtb": "EBTB (or TBTB) stands for tracheobronchial tuberculosis, a subtype of tuberculosis.\nYou can learn more about it in our wiki at https://ai-ebtb.com/ebtb-wiki.html",
    "ai-ebtb": "AI-EBTB aims to quickly diagnose EBTB with high precision.\nWe are a professional team with a large, high-quality database to train our model.\nLearn more about our team at https://ai-ebtb.com/about-us.html",
    "collaborate": "If you are interested in collaborating with us, check out https://ai-ebtb.com/collaborate-with-us.html for more details.",
    "general-issues": "I'm sorry if you are having problems using our service.\nYou can contact us by emailing enquiries@ai-ebtb.com for all kinds of problems.",
    "no-otp": "It seems that you are having problems receiving verification codes.\nPlease wait for a few minutes, and check your spam folder.\nIf that doesn't work, contact us by emailing enquiries@ai-ebtb.com and we will sort out the problem for you.",
    "service-issues": "We are sorry that you are having problems with the diagnosis service.\nPlease take a screenshot of the problem and email us at enquiries@ai-ebtb.com.\nWe will do our best to sort out the problem as fast as possible.",
    "wiki-issues": "We are sorry that your experience with our wiki isn't great.\nWe are actively working on adding more content to the wiki,\nplease understand that we are a small team of six.",
    "responsive-issues": "We are sorry that your experience with our website isn't great.\nOur website is intended to be used on desktop,\nand even though we have designed it to be responsive,\nit might not look great on smaller devices.",
    "translation-issues": "We are sorry that our website's translations don't meet your standards.\nYou can email us at enquiries@ai-ebtb.com to leave your feedback and help us improve the website.",
    "navigation-issues": "We are sorry that you are having trouble navigating the website.\nYou can email us at enquiries@ai-ebtb.com to leave your feedback and help us improve the website.",
    "accessibility-issues": "We are sorry that currently our website is not designed with accessibility in mind.\nIn the future this will be improved.\nYou can email us at enquiries@ai-ebtb.com to leave your feedback and help us improve the website.",
    "copyright-issues": "We are sorry if our website is violating your copyright.\nEmail us at enquiries@ai-ebtb.com and we will sort out the problem immediately.",
    "legal-issues": "Please understand that we are a small team of six and have no legal experts.\nIf you are having any legal issues, email us at enquiries@ai-ebtb.com and we will try our best to sort out the problem.",
    "contact-issues": "If you are having trouble with contacting us, we sincerely apologize.\nPlease be patient and we will get back to you soon.",
    "login-issues": "If you are having trouble logging in,\nsend us a screenshot and a description of the problem at enquiries@ai-ebtb.com.\nWe will sort out the problem as soon as possible.",
    "account-issues": "If you wish to change your account nickname/email address, or are having other difficulties with your account,\nemail us at enquiries@ai-ebtb.com and we will sort out the problem for you.",
    "password-issues": "If you wish to change your password, you can do so in dashboard > account settings > change password.\nIf you've forgotten your password, you can reset it at the login page.",
    "request-human": "I'm sorry if you are not satisfied with my responses.\nWe currently do not have real human online chat service.\nYou can email us at enquiries@ai-ebtb.com to talk about anything.",
    "compliment": "Thank you! I'm very glad that you are satisfied with my service. ;)", 
    "criticize": "I'm sorry that you are not satisfied with my service.\nI'm trying my best, but you can email us at enquiries@ai-ebtb.com to get better service.",
    "apologize": "Thank you for being polite. Always my pleasure to serve you ;)",
    "about-chatbot": "I'm a chatbot created to assist you.\nThere's not much to say about me.\nI'm always happy to serve you ;)"
}

# Define verbatim words and phrases
verbatim_phrases = {"AI-EBTB", "EBTB", "TBTB"}

# Text Preprocessing
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Lemmatization
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    # Spelling correction
    corrected = str(TextBlob(lemmatized).correct())
    return corrected

def extract_verbatim_phrases(text, verbatim_phrases):
    placeholders = {}
    for phrase in verbatim_phrases:
        if phrase in text:
            placeholder = f"VERBATIM_{hash(phrase)}"
            placeholders[placeholder] = phrase
            text = text.replace(phrase, placeholder)
    return text, placeholders

def reintegrate_verbatim_phrases(text, placeholders):
    for placeholder, phrase in placeholders.items():
        text = text.replace(placeholder, phrase)
    return text

# Function to recognize intent with verbatim and semantic similarity
def recognize_intent(text):
    # Extract verbatim phrases
    text, placeholders = extract_verbatim_phrases(text, verbatim_phrases)
    
    preprocessed_text = preprocess(text)
    # Reintegrate verbatim phrases after preprocessing
    preprocessed_text = reintegrate_verbatim_phrases(preprocessed_text, placeholders)
    
    words = preprocessed_text.split()
    
    # Check for verbatim phrases
    for phrase in verbatim_phrases:
        if phrase in text:
            if phrase == "AI-EBTB":
                return "ai-ebtb"
            elif phrase == "EBTB" or phrase == "TBTB":
                return "about_ebtb"

    # Initialize intent scores
    intent_scores = {intent: 0 for intent in responses.keys()}

    # Minimum similarity threshold
    min_similarity_threshold = 0.65

    print("\n--- Debugging Information ---")
    print(f"Input Text: {text}")
    print(f"Preprocessed Text: {preprocessed_text}")
    
    # Calculate semantic similarity for each word and aggregate scores
    for word in words:
        if word in model:
            print(f"\nWord: {word}")
            for intent in intent_scores:
                # Get similar words for the current intent
                similar_words = words_by_intent[intent]
                similarities = []
                for sim_word in similar_words:
                    if sim_word in model:
                        similarity = model.similarity(word, sim_word)
                        similarities.append(similarity)
                        print(f"Similarity between '{word}' and '{sim_word}': {similarity:.4f}")
                
                if similarities:
                    max_similarity = max(similarities)
                    if max_similarity >= min_similarity_threshold:
                        intent_scores[intent] += max_similarity
                        print(f"Intent: {intent}, Max Similarity for Word '{word}': {max_similarity:.4f}, Updated Score: {intent_scores[intent]:.4f}")
                    else:
                        print(f"Intent: {intent}, Max Similarity for Word '{word}' below threshold: {max_similarity:.4f}")
                else:
                    print(f"Intent: {intent}, No Similarity Found for Word '{word}'")
    
    # Print the final scores for all intents
    print("\nFinal Intent Scores:")
    for intent, score in intent_scores.items():
        print(f"{intent}: {score:.4f}")

    # Select the intent with the highest score
    predicted_intent = max(intent_scores, key=intent_scores.get)
    print(f"Predicted Intent: {predicted_intent}")
    print("--- End of Debugging Information ---\n")
    
    # Apply a threshold to ensure a certain level of confidence
    if intent_scores[predicted_intent] > 0.7:  # You can adjust this threshold
        return predicted_intent
    else:
        return "unknown"

# Function to process text
def process_text(text):
    # language = detect(text) # just assume input is english
    doc = nlp(text)
    tokens = [token.text for token in doc]
    intent = recognize_intent(text)
    
    return tokens, intent

# Function to generate response
def generate_response(intent):
    return responses.get(intent, "I'm sorry, I didn't understand that.")


# WebSocket handler
async def handle_message(websocket, path):
    async for message in websocket:
        intent = recognize_intent(message)
        response = responses.get(intent)
        await websocket.send(response)

# Start WebSocket server
start_server = websockets.serve(handle_message, "0.0.0.0", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
