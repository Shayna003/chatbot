import asyncio
import websockets
import spacy
import jieba
import pickle
from gensim.models import KeyedVectors

print("loading spacy")
# Load SpaCy model for Chinese lemmatization
nlp = spacy.load('zh_core_web_sm')

print("loading cc.zh.300.vec")
# Load pre-trained FastText embeddings for Chinese
model = KeyedVectors.load_word2vec_format('cc.zh.300.vec', binary=False)

print("loading pipeline")
# Load the pipeline, label encoder, and words_by_intent
with open('model_zh/pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

print("loading encoder")
with open('model_zh/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("loading words by intent")
with open('model_zh/words_by_intent.pkl', 'rb') as f:
    words_by_intent = pickle.load(f)

# Define responses using a dictionary (translated to Chinese)
responses = {
    "greeting": "你好！今天我能帮你做什么？ :)",
    "ask_name": "我叫小影，是一个聊天机器人，很高兴为您服务～ :)",
    "farewell": "再见！祝你有美好的一天！ ;)",
    "about_ebtb": "EBTB（或TBTB）代表气管支气管结核，是结核病的一个亚型。\n你可以在我们的百科上了解更多信息：https://ai-ebtb.com/ebtb-wiki.html",
    "ai-ebtb": "AI-EBTB旨在快速、高精度地诊断EBTB。\n我们是一个专业团队，拥有大量高质量的数据库来训练我们的模型。\n在https://ai-ebtb.com/about-us.html了解更多关于我们的信息。",
    "collaborate": "如果你有兴趣与我们合作，请查看https://ai-ebtb.com/collaborate-with-us.html了解更多详情。",
    "general-issues": "很抱歉你在使用我们的服务时遇到问题。\n你可以通过发送邮件至enquiries@ai-ebtb.com与我们联系各种问题。",
    "no-otp": "看起来你在接收验证码时遇到了问题。\n请等待几分钟，检查你的垃圾邮件文件夹。\n如果仍然没有收到，请发送邮件至enquiries@ai-ebtb.com，我们会为你解决问题。",
    "service-issues": "我们很抱歉你在使用诊断服务时遇到了问题。\n请发送问题截图到enquiries@ai-ebtb.com，我们会尽快解决问题。",
    "wiki-issues": "我们很抱歉你的百科体验不佳。\n我们正在积极添加更多内容到百科，请理解我们是一个由六人组成的小团队。",
    "responsive-issues": "我们很抱歉你对我们网站的体验不佳。\n我们的网站是为桌面设计的，尽管我们设计了响应式，但在较小设备上可能效果不佳。",
    "translation-issues": "我们很抱歉我们网站的翻译不符合你的标准。\n你可以发送邮件至enquiries@ai-ebtb.com，留下你的反馈，帮助我们改进网站。",
    "navigation-issues": "我们很抱歉你在浏览网站时遇到问题。\n你可以发送邮件至enquiries@ai-ebtb.com，留下你的反馈，帮助我们改进网站。",
    "accessibility-issues": "我们很抱歉目前我们的网站没有无障碍设计。\n未来我们会改进这一点。\n你可以发送邮件至enquiries@ai-ebtb.com，留下你的反馈，帮助我们改进网站。",
    "copyright-issues": "我们很抱歉如果我们的网站侵犯了你的版权。\n发送邮件至enquiries@ai-ebtb.com，我们会立即解决问题。",
    "legal-issues": "请理解我们是一个由六人组成的小团队，没有法律专家。\n如果你遇到任何法律问题，请发送邮件至enquiries@ai-ebtb.com，我们会尽力解决问题。",
    "contact-issues": "如果你在联系时遇到问题，我们真诚地道歉。\n请耐心等待，我们会尽快回复你。",
    "login-issues": "如果你在登录时遇到问题，请发送问题截图和描述到enquiries@ai-ebtb.com。\n我们会尽快解决问题。",
    "account-issues": "如果你想更改账户昵称/邮箱地址，或遇到其他账户问题，\n请发送邮件至enquiries@ai-ebtb.com，我们会为你解决问题。",
    "password-issues": "如果你想更改密码，可以在dashboard > account settings > change password进行更改。\n如果你忘记了密码，可以在登录页面重置。",
    "request-human": "很抱歉如果你对我的回答不满意。\n我们目前没有真人在线聊天服务。\n你可以发送邮件至enquiries@ai-ebtb.com与我们讨论任何问题。",
    "compliment": "谢谢！很高兴你对我的服务满意。 ;)", 
    "criticize": "很抱歉你对我的服务不满意。\n我在尽力而为，但你可以发送邮件至enquiries@ai-ebtb.com获取更好的服务。",
    "apologize": "谢谢你的礼貌。很高兴为你服务 ;)",
    "about-chatbot": "我叫小影，是一个聊天机器人。\n关于我没有太多可说的。\n我很高兴为您服务～"
}

# Define verbatim words and phrases
verbatim_phrases = {"AI-EBTB", "EBTB", "TBTB"}

# Text Preprocessing
def preprocess(text):
    # Use jieba for Chinese tokenization
    text = " ".join(jieba.cut(text))
    return text

# Function to recognize intent with verbatim and semantic similarity
def recognize_intent(text):
    # Check for verbatim phrases
    for phrase in verbatim_phrases:
        if phrase in text:
            if phrase == "AI-EBTB" or phrase == "ai-ebtb":
                return "ai-ebtb"
            elif phrase == "EBTB" or phrase == "ebtb" or phrase == "TBTB" or phrase == "tbtb":
                return "about_ebtb"

    preprocessed_text = preprocess(text)
    
    words = preprocessed_text.split()

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
    intent = recognize_intent(text)
    tokens = list(jieba.cut(text))
    return tokens, intent

# Function to generate response
def generate_response(intent):
    return responses.get(intent, "对不起，我不明白你的意思。")

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