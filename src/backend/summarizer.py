import json
# import torch
import uuid
from typing import  List


import warnings

from transformers import pipeline


"""
This script provides functionalities for setting up and interacting with the ModelClass-13b language model (LLM).  

Classes:
- ModelClass: A class responsible for loading the ModelClass-13b model using Hugging Face Transformers. 
          It includes a method to generate responses from the LLM, as well as a generator function to output streaming responses.

"""

def uuid_factory() -> str:
    """
    Generates and returns a UUID (Universally Unique Identifier).

    Returns:
        str: A unique identifier in the form of a string.
    """
    return str(uuid.uuid1())

class ModelClass:

    MODEL_PATH: str = "sshleifer/distilbart-cnn-12-6"
    MODEL_EOS_TOKENS_IDS: List[int] = [2]

    
    def __init__(self):
        """
        Initializes an instance of the ModelClass class and loads the ModelClass-13b model.
        """

        self.__load_model__()
        self.summarizer_pipeline
       

    def __load_model__(self):
        """
        Loads the model and tokenizer.

        Raises:
            Exception: If the model fails to load.
        """
        print('\n\nLoading the model, this might take a while...')
        
        
        try:
            #assert torch.cuda.is_available()

            self.model_name = self.MODEL_PATH
            self.summarizer_pipeline = pipeline(
                task = "summarization",
                    model= self.model_name,
                )

        except Exception as e:
            raise(Exception([f"Failed to load the {self.model_name} model, this was cause by the folowing exception: ", e]))    
        else:
            print(f"Model {self.model_name} loaded successfully")
   
    def summarize_document(self, document) -> dict: 
        """_summary_
        """
        summary_id = uuid_factory() #uuid1() # ToDo: check if uuid is in db, if not create it
        summary_pipeline_outputs = self.summarizer_pipeline(document, truncation=True, batch_size=1)
        new_db_entry = {'uuid': summary_id, 'document': document, 'summary': summary_pipeline_outputs[0]['summary_text'],}
        
        return new_db_entry
