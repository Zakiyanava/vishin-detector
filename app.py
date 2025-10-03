import speech_recognition as sr
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now (OTP-related call)...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except:
            print("⚠️ Could not recognize speech.")
            return ""

def detect_vishing(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    return prediction[0]

if __name__ == "__main__":
    spoken_text = record_audio()
    if spoken_text:
        result = detect_vishing(spoken_text)
        if result == "vishing":
            print("🚨 WARNING: Vishing attempt detected!")
        else:
            print("✅ This message seems safe.")
