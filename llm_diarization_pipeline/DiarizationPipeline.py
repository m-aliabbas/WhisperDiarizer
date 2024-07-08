from llm_diarization_pipeline.GroqTranscriber import GroqTranscriber
from llm_diarization_pipeline.LLMDiarizer import LLMDiarizer
import os
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class DiarizationPipeline:
    def __init__(self,prompt,out_model=None) -> None:
        self.transcriber = GroqTranscriber()
        self.diarizer = LLMDiarizer(prompt=prompt,out_model=out_model)

    def get_transcription_result(self,audio_file):
        return self.transcriber.get_transcription_result(audio_file)

    def diarize(self,conversation):
        return self.diarizer.diarize(conversation)
    
    def forward(self,audio_file,out_path='./output/'):
        try:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            base_filename = os.path.basename(audio_file)[:-4]
            print("-"*30)
            print("FILE NAME: ",base_filename)
            output_file_path = os.path.join(out_path,base_filename+'.json')
            print("Generation Transcript")
            text,conversation = self.get_transcription_result(audio_file)
            print(conversation)
            print("Diarizing Transcript")
            diarization_results = self.diarize(conversation)
            print("Saving Result")
            print("-"*30)
            json_output = {'file_name':base_filename,'text':text,'diarization_results':diarization_results}
            with open(output_file_path, 'w') as f:
                json.dump(json_output, f)
            return json_output
        except Exception as e:
            print('='*30)
            print("FILE NAME: ",audio_file)
            print("ERROR: ",e)
            print("="*30)
    
if __name__ == "__main__":  
    audio_file = 'example_audios/20230320-100607_6185701318-all.wav'
    prompt = "And this is a conversation between Seller and Customer. Seller is selling insurance or something like that. There can be more than one sellers in conversation."
    class ConversationObject(BaseModel):
        id: str
        start: float
        end: float
        text: str
        speaker: str
        class_label: str = Field(description="""classify the text into one of following classes
                                      answering_machine = 'recording running on phone numbers'
                                      interested = 'customer showing some interest'
                                      'dnc' = 'customer asking for do not call me again'
                                      'busy' = 'customer is saying that he is in something'
                                      'callback' = 'customer ask to call back'
                                      'not_interested' = 'customer not interested in anything'
                                      'already' = 'customer is already having offer'
                                      'affirmation' = 'customer agree to go forward'
                                      'decline' = 'customer disagree to go forward'
                                       'weather query' = 'customer asking about weather'
                                      'location query' = 'customer asking about location'
                                      'email query' = 'customer is asking about can we email him'
                                      'transfer request' = 'customer asking us to transfer call to senior',
                                       'other' = 'something in not above'
                                      """)
    class ConversationList(BaseModel):
        conversations: List[ConversationObject]
    model = DiarizationPipeline(prompt=prompt,out_model=ConversationList)
    results = model.forward(audio_file)