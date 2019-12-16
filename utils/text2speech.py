from data_io.paths import get_tmp_dir
from gtts import gTTS
import os
import subprocess

def t2s(text):
    tts = gTTS(text=text, lang="en")
    mp3path = os.path.join(get_tmp_dir(), "text.mp3")
    os.makedirs(get_tmp_dir(), exist_ok=True)
    tts.save(mp3path)

def say(text, dontblock=False):
    t2s(text)
    repeat(dontblock)

def repeat(dontblock=False):
    mp3path = os.path.join(get_tmp_dir(), "text.mp3")
    FNULL = open(os.devnull, 'w')
    subprocess.Popen("mpg321 " + mp3path + (" &" if dontblock else ""), shell=True, stderr=FNULL, stdout=FNULL)