from faster_whisper import WhisperModel
from config import diarizer_config 

class Transcriber(object):
    def __init__(self) -> None:
        self.model = WhisperModel(diarizer_config['transcriptor']['whisper_model'], 
                             compute_type=diarizer_config['transcriptor']['compute_type'])
        self.options = dict(language=diarizer_config['transcriptor']['lang'], beam_size=5, best_of=5)
        self.transcribe_options = dict(task="transcribe", **self.options)
        
    def get_segments(self,audio_file):
        segments_raw, info = self.model.transcribe(audio_file, **self.transcribe_options)
        segments = []
        i = 0
        for segment_chunk in segments_raw:
            
            chunk = {}
            chunk["start"] = segment_chunk.start
            chunk["end"] = segment_chunk.end
            chunk["text"] = segment_chunk.text
            segments.append(chunk)
            i += 1
        
        print(" [+] transcribe audio done with fast whisper ")
        return segments
