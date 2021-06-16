#put  pip install gtts and pyttsxpip install pyttsx3==2.71 before running

import pyttsx3
import gtts

def texttospeech(text):
    
    engine = pyttsx3.init()
    tts = gtts.gTTS("Hello world")
    engine.say(text)
    # play the speech
    engine.runAndWait()
    return tts



def example():
    texttospeech("This is a test message to test the program")



