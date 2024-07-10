from server_diarization.client import DiarizationPipeline
from utils.utils import *
import glob


output_folder = "./output/"
server_url = "http://148.251.195.218:9090/uploadfile/"



if __name__ == "__main__":


    # please write your input folder name here
    # line 12
    USER_INPUT_FOLDER = 'AM'
    file_names = glob.glob(f"./input_audios/{USER_INPUT_FOLDER}/*.mp3")
 
    # prompt = "And this is a conversation between Seller and Customer. Seller is selling insurance or something like that. There can be more than one sellers in conversation."
    model = DiarizationPipeline(server_uri=server_url)
    
    for file_name in file_names:
        results = model.forward(file_name,output_folder=output_folder,server_url=server_url)
    