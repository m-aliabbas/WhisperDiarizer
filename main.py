from WhisperDiarizer import WhisperDiarizer

model = WhisperDiarizer(num_speaker=2)
import timeit

start = timeit.default_timer()
df,result=model.diarize('example_audios/clear.wav')
end = timeit.default_timer()
print(result)
print(start-end)



