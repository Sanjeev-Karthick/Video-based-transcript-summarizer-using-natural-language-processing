
import moviepy.editor as mp
my_clip = mp.VideoFileClip(r"C:\Users\vijayalakshmi\PycharmProjects\offline_vid_to_aud\What is Technology.mp4")
print(my_clip)
my_clip.audio.write_audiofile(r"C:\Users\vijayalakshmi\PycharmProjects\offline_vid_to_aud\my_result3.wav")
#Python 2.x program to transcribe an Audio file
'''
import speech_recognition as sr

AUDIO_FILE = 'C:/Users/vijayalakshmi/PycharmProjects/offline_vid_to_aud/my_result3.wav'

# use the audio file as the audio source

r = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:
	#reads the audio file. Here we use record instead of
	#listen
	audio = r.record(source)

try:
	print("The audio file contains: " + r.recognize_google(audio))

except sr.UnknownValueError:
	print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
	print("Could not request results from Google Speech Recognition service; {0}".format(e))'''
# importing libraries
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# create a speech recognition object
r = sr.Recognizer()

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text
path = "C:/Users/vijayalakshmi/PycharmProjects/offline_vid_to_aud/my_result3.wav"
print("\nFull text:", get_large_audio_transcription(path))
