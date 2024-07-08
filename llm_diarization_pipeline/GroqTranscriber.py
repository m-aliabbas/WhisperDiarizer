from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()



class GroqTranscriber:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def get_transcription_result(self,audio_file):
        '''
        arguments: 
        returns text and segments from audio file
        '''
        with open(audio_file, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
            file=(audio_file, file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            )
            
            return transcription.text,transcription.segments


if __name__ == '__main__':
    model = GroqTranscriber()
    transcription,seg = model.get_transcription_result('../example_audios/clear.wav')
    print(transcription,seg)
    





      