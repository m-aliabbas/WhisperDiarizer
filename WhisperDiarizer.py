from SpeakerClustering import SpeakerClustering
from SpeakerEmbedding import SpeakerEmbedder
from Transcriber import Transcriber
import pandas as pd
from utils import convert_time
import timeit
class WhisperDiarizer(object):
    def __init__(self,num_speaker=2) -> None:
        self.num_speaker=num_speaker
        self.transcriber = Transcriber()
        self.speaker_embedder = SpeakerEmbedder()
        self.speaker_clustering = SpeakerClustering(num_speaker=self.num_speaker)

        self.segments = []
        self.audio_file = None
        self.embeddings = None

    def diarize(self,audio_file):
        self.audio_file= audio_file
        t1 = timeit.default_timer()
        self.segments = self.transcriber.get_segments(audio_file=audio_file)
        t2 = timeit.default_timer()

        print('[-] Whisper taken',t2-t1)

        t3 = timeit.default_timer()
        self.embeddings = self.speaker_embedder.get_embeddings(segments=self.segments,
                                                               audio_file=audio_file)
        t4 = timeit.default_timer()
        print('[-] Embedding taken',t4-t3)
        t5 = timeit.default_timer()
        self.clustered_segments = self.speaker_clustering.assign_speaker_label(embeddings=self.embeddings,
                                                                               segments=self.segments)
        t6 = timeit.default_timer()
        print('[-] Clustering taken',t5-t6)
        df_results = pd.DataFrame(self.clustered_segments)
        return df_results, self.clustered_segments
        
        