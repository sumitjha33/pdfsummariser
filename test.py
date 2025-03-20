import pygame
import time
import os
from gtts import gTTS
from threading import Thread

# PDF Summarizer Output (Example)
summary_text = "Hello! This is a sample summary from the PDF summarizer."

# Convert text to speech using gTTS
def generate_audio(text, filename="summary_audio.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)

# Function to play audio
def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

# Function to display text in a typewriter effect
def typewriter_effect(text, delay=0.2):
    for word in text.split():
        print(word, end=" ", flush=True)
        time.sleep(delay)
    print("\n")

# Generate the speech file
generate_audio(summary_text)

# Run text display and audio playback in sync
def run_sync():
    # Start playing the audio in a separate thread
    audio_thread = Thread(target=play_audio, args=("summary_audio.mp3",))
    audio_thread.start()

    # Display typewriter text with a matching delay
    typewriter_effect(summary_text, delay=0.4)  # Adjust delay if needed

    # Wait until audio finishes
    audio_thread.join()

# Run the program
run_sync()

# Cleanup the audio file after execution (optional)
os.remove("summary_audio.mp3")
