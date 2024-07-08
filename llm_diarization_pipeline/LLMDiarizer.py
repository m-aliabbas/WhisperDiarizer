from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

load_dotenv()

class ConversationObject(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str = ''
    class_label: str = Field(description="""classify the text into one of the following classes:
        answering_machine = 'recording running on phone numbers'
        interested = 'customer showing some interest'
        dnc = 'customer asking for do not call me again'
        busy = 'customer is saying that he is in something'
        callback = 'customer ask to call back'
        not_interested = 'customer not interested in anything'
        already = 'customer is already having offer'
        affirmation = 'customer agree to go forward'
        decline = 'customer disagree to go forward'
        weather_query = 'customer asking about weather'
        location_query = 'customer asking about location'
        email_query = 'customer is asking about can we email him'
        transfer_request = 'customer asking us to transfer call to senior'
        other = 'something not above'
    """)

class ConversationList(BaseModel):
    conversations: List[ConversationObject]

class LLMDiarizer:
    def __init__(self, prompt, out_model) -> None:
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
        )
        self.about_conversation = prompt
        self.parser = JsonOutputParser(pydantic_object=out_model)
        
        diarization_prompt = """
            You are a diarization agent. Also, classify the text chunks to different classes. 
            If there is only one speaker in conversation, classify them as 'Customer'.
            If something is like the following, it will be treated as an answering machine:
            ```
            Thank you for calling. I'm sorry I'm unable to answer the telephone at the moment, but leave your name and number, and I will get right back with you as soon as possible.
            ```
            about conversation: {about_conversation}
            conversation: {conversation}
            Your task is to perform speaker diarization and classify each chunk.
            Output format JSON should be like this:
            conversation: []
            Answer:
        """
        
        self.diarization_assistant_prompt_template = PromptTemplate(
            input_variables=["about_conversation", "conversation"],
            template=diarization_prompt
        )
        
        self.llm_chain = self.diarization_assistant_prompt_template | self.llm | self.parser

    def diarize(self, conversation):
        conversation = self.clean_conversation(conversation)
        if len(conversation) == 1:
            conversation[0]['speaker'] = 'Customer'
            conversation[0]['class_label'] = self.classify_text(conversation[0]['text'])
            return conversation

        llm_response = self.llm_chain.invoke({"about_conversation": self.about_conversation, "conversation": conversation})
        return llm_response

    def classify_text(self, text):
        # Add your custom classification logic here or use the LLM to classify
        return 'answering_machine' if 'leave a message' in text.lower() else 'other'

    def clean_conversation(self, conversation):
        cleaned_conversation = []
        required_keys = {'id', 'start', 'end', 'text'}
        for segment in conversation:
            try:
                cleaned_segment = {key: segment[key] for key in required_keys if key in segment}
                if len(cleaned_segment) == len(required_keys):
                    cleaned_conversation.append(cleaned_segment)
                else:
                    print(f"Skipping segment {segment['id']} due to missing keys.")
            except KeyError as e:
                print(f"Key error: {e} in segment {segment}. Skipping this segment.")
            except Exception as e:
                print(f"Unexpected error: {e} in segment {segment}. Skipping this segment.")
        return cleaned_conversation

if __name__ == '__main__':
    conversation = [{'id': 0, 'seek': 0, 'start': 0, 'end': 10.08, 'text': ' The U-Mail subscriber at 3185129442 is currently unavailable.', 'tokens': [50365, 440, 624, 12, 44, 864, 26122, 412, 805, 6494, 20, 4762, 24, 13912, 17, 307, 4362, 36541, 32699, 13, 50869], 'temperature': 0, 'avg_logprob': -0.23588958, 'compression_ratio': 1.1206896, 'no_speech_prob': 0.09667625}, {'id': 1, 'seek': 0, 'start': 10.4, 'end': 11.78, 'text': ' Please speak clearly.', 'tokens': [50885, 2555, 1710, 4448, 13, 50954], 'temperature': 0, 'avg_logprob': -0.23588958, 'compression_ratio': 1.1206896, 'no_speech_prob': 0.09667625}, {'id': 2, 'seek': 0, 'start': 12.16, 'end': 15.4, 'text': ' Your voicemail is being transcribed by U-Mail.', 'tokens': [50973, 2260, 1650, 46343, 864, 307, 885, 1145, 18732, 538, 624, 12, 44, 864, 13, 51135], 'temperature': 0, 'avg_logprob': -0.23588958, 'compression_ratio': 1.1206896, 'no_speech_prob': 0.09667625}]
    
    llm_diarizer = LLMDiarizer(
        prompt="And this is a conversation between Seller and Customer. Seller is selling insurance or something like that. There can be more than one sellers in conversation. Sometimes Seller or Customer either of them won't be in conversation. Like a one-sided conversation. Then just return it as Seller.",
        out_model=ConversationList
    )

    result = llm_diarizer.diarize(conversation=conversation)
    print('Results: ', result)
