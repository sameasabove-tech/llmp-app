import json
# import torch
from typing import  List
import pandas as pd

import warnings

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from src.backend.utlities.util_fns import uuid_factory     


"""
This script provides functionalities for setting up and interacting with the xxxxx language model (LLM).  

Classes:
- ModelClass: A class responsible for loading the ModelClass-13b model using Hugging Face Transformers. 
          It includes a method to generate responses from the LLM, as well as a generator function to output streaming responses.

"""

# work flow: 
# 1) creating a file system hiearchy and then embedding the document. For example: --> test this with "import os" first
# Desktop
#   L__School
#       L__Course_150
#           L__Chapter1.txt
#           L__Chapter2.txt
#           L__RQ-Chapter1.pdf
#           L__RQ-Chapter2.pdf
#   L__Personal
#       L__passport.jpeg
#       L__doc_of_Ls.txt
# Data table would be:
# document_contents |    docuemnt_name  | Desktop |  School  | Course_150 |  Personal  |
#      xxxxxxx      | Chapter1.txt      |    1    |    1     |    1       |     0      |
#      xxxxxxx      | Chapter2.txt      |    1    |    1     |    1       |     0      |
#      xxxxxxx      | RQ-Chapter1.pdf   |    1    |    1     |    1       |     0      |
#      xxxxxxx      | RQ-Chapter2.pdf   |    1    |    1     |    0       |     0      |
#      xxxxxxx      | passport.jpeg     |    1    |    0     |    0       |     1      |
#      xxxxxxx      | doc_of_Ls.txt     |    1    |    0     |    0       |     1      |

# 2) Go through each file contents and embedd each document
# 3) learning model to classify document into a system

# Class Methods:
# 1. load model
# 2. document_tree_data_preprocess, csv for testing at beginning 
# 3. text_embedding
# 4. oneshot learning pipeline
# 5. better learning model? ie. one that uses the tree hiearchy? code later and build these pipelines instead for now


class OrganizerClass:
    MODEL_PATH_TXT_EMB: str = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_EOS_TOKENS_IDS: List[int] = [2]

    
    def __init__(self):
        """
        Initializes an instance of the ModelClass class and loads the ModelClass-13b model.
        """

        self.__load_model__()
        # self.summarizer_pipeline
       

    def __load_model__(self):
        """
        Loads the model and tokenizer.

        Raises:
            Exception: If the model fails to load.
        """
        print('\n\nLoading the model, this might take a while...')
        
        
        try:
            #assert torch.cuda.is_available()

            self.model_name_txt_emb = self.MODEL_PATH_TXT_EMB
            self.model_txt_emb = SentenceTransformer(self.model_name_txt_emb)

        except Exception as e:
            raise(Exception([f"Failed to load the {self.model_name_txt_emb} model, this was cause by the folowing exception: ", e]))    
        else:
            print(f"Model {self.model_name_txt_emb} loaded successfully")
   
    def encode_document(self, document): 
        """Document data will be transformed to a text emedding and stored in the text_embedding variable of the dataclass.
        """
        encodings = self.model_txt_emb.encode(document)
        text_embedding_list = [(uuid_factory(), e) for e in encodings]
        return text_embedding_list

