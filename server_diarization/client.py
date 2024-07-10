import os
import requests
import json


class DiarizationPipeline(object):
    def __init__(self,server_uri) -> None:
        pass


    def send_file_to_server(self,file_path, server_url):
        # Open the file in binary mode
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(server_url, files=files)
            return response.json()

    def forward(self,filename, output_folder, server_url):
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Iterate through all MP3 files in the input folder
        
        if filename.endswith(".mp3"):
            file_path = filename
            print('='*30)
            print(f'Processing file:', file_path)
            # Send the file to the server and get the JSON response
            json_response = self.send_file_to_server(file_path, server_url)
            
            file_name_base = os.path.basename(file_path)
            # Save the JSON response to the output folder
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name_base)[0]}.json")
            with open(output_file_path, "w") as output_file:
                json.dump(json_response, output_file, indent=4)
            print('Saved to:', output_file_path)
            print('='*30)

