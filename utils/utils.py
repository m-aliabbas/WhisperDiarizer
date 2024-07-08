from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

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
    