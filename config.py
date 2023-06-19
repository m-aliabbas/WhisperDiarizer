diarizer_config = dict()

diarizer_config['transcriptor'] = dict()
diarizer_config['transcriptor']['whisper_model'] = 'base'
diarizer_config['transcriptor']['compute_type'] = 'int8'
diarizer_config['transcriptor']['lang'] = 'en'


diarizer_config['embedding_model'] = dict()
diarizer_config['embedding_model']['model_name']="speechbrain/spkrec-ecapa-voxceleb"
diarizer_config['embedding_model']['embedding_dim']=192