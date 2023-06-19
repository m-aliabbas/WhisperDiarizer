from faster_whisper import WhisperModel
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import torch
import numpy as np
from config import diarizer_config
class SpeakerEmbedder(object):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = PretrainedSpeakerEmbedding(
                    diarizer_config['embedding_model']['model_name'],
                    device=self.device)
        
    def get_embeddings(self,segments,audio_file):
        
        self.embeddings = np.zeros(shape=(len(segments), 
                                          diarizer_config['embedding_model']['embedding_dim']))
        for i, segment in enumerate(segments):
            self.embeddings[i] = self.__segment_embedding(segment,audio_file=audio_file)
        self.embeddings = np.nan_to_num(self.embeddings)
        return self.embeddings

    def __segment_embedding(self,segment,audio_file):
        audio = Audio()
        start = segment["start"]
        end =segment["end"]-1000 #whisper overshoot last sometime
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(audio_file, clip)
        waveform = waveform[0].unsqueeze(0)
        waveform = waveform.unsqueeze(0)
        return self.embedding_model(waveform)


