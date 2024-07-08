from llm_diarization_pipeline.DiarizationPipeline import DiarizationPipeline
from utils.utils import *
import glob



if __name__ == "__main__":


    # please write your input folder name here
    # line 12
    USER_INPUT_FOLDER = ''
    file_names = glob.glob(f"/input/{USER_INPUT_FOLDER}/*.mp3")
 
    prompt = "And this is a conversation between Seller and Customer. Seller is selling insurance or something like that. There can be more than one sellers in conversation."
    model = DiarizationPipeline(prompt=prompt,out_model=ConversationList)
    
    first50_files = file_names[:50]
    for file_name in first50_files:
        results = model.forward(file_name)
    