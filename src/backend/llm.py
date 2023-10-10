import json
import torch

from threading import Thread
from typing import  List,  Generator

import warnings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


"""
This script provides functionalities for setting up and interacting with the ModelClass-13b language model (LLM).  

Classes:
- ModelClass: A class responsible for loading the ModelClass-13b model using Hugging Face Transformers. 
          It includes a method to generate responses from the LLM, as well as a generator function to output streaming responses.

"""

class ModelClass:

    MODEL_PATH: str = "google/flan-t5-small"
    MODEL_EOS_TOKENS_IDS: List[int] = [2]

    
    def __init__(self):
        """
        Initializes an instance of the ModelClass class and loads the ModelClass-13b model.
        """

        self.__load_model__()
       

    

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
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
            #self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_PATH)
            self.generation_config = GenerationConfig.from_pretrained(self.MODEL_PATH)
            self.generation_config.update(do_sample = True, max_length = 1000)

            
        
        except Exception as e:
            raise(Exception([f"Failed to load the {self.model_name} model, this was cause by the folowing exception: ", e]))    
        else:
            print(f"Model {self.model_name} loaded successfully")
   
    def ask_llm(self, question: str, debug:bool = False, **kwargs) -> tuple:
        """
        Generates a response from the ModelClass model given a question. Not using the pipeline to provide warnings and  debug information to the user.

        Parameters:
            question (str): The input question.
            debug (bool): If True will provide additional debug info about the prompt 
            **kwargs: Additional keyword arguments.

        Returns:
            tuple:  The generated LLM response, a list of warnings, dict containg some debug info
        """
        with warnings.catch_warnings(record=True) as warnings_list: # Collecting all the warnings so they can be passed to the API caller
            
            inputs =  self.tokenizer(question, return_tensors='pt')
            eos_token_id = kwargs.pop('eos_token_id', self.MODEL_EOS_TOKENS_IDS)
            generation_config = GenerationConfig(**self.generation_config.to_diff_dict())
            generation_config.update(eos_token_id=eos_token_id, **kwargs)
            outputs_encoded= self.model.generate(**inputs, do_sample = True, generation_config=generation_config ).to('cpu')
            generated = self.tokenizer.batch_decode(outputs_encoded, skip_special_tokens=True)
            
        
        warning_messages = ' /n'.join([warn.message.__str__().strip() for warn in warnings_list])
        warnings.warn(warning_messages)
        return (generated[0], # Returning only the last assistant response
                warning_messages,
                 {'input_len': inputs['input_ids'].shape[-1] ,
                   'inputs': self.tokenizer.batch_decode(inputs['input_ids']) ,
                   'output_full_len' : outputs_encoded.shape[-1],
                   'generation_config': generation_config.to_dict() } if debug else {}
         ) 
   
    class StopOnTokens(StoppingCriteria):
        """ Class needed for the streaming generation """
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [29, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
        
    def ask_llm_stream(self, question: str, **kwargs) -> Generator[str, None, None]:
        """ 
        Returns a Generator providing the response from the llm in a streaming manner 
        
         Parameters:
            question (str): The input question
            **kwargs: Additional keyword arguments.

        Returns:
            generator:  The generated LLM response
        
        """
        stop = self.StopOnTokens()
        model_inputs = self.tokenizer(question, return_tensors='pt')
        # Setting up the streamer on a separate Thread to fetch words in a non blocking way, 
        # see for more details : https://huggingface.co/docs/transformers/v4.31.0/en/internal/generation_utils#transformers.TextIteratorStreamer 

        streamer = TextIteratorStreamer(self.tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generation_config = GenerationConfig(**self.generation_config.to_diff_dict())
        generation_config.update(**kwargs)
        generation_config.update(num_beams = 1) # Mandatory for streaming generation, overriding any user settings 
        generate_kwargs = dict(
                                model_inputs,
                                streamer=streamer,
                                generation_config = generation_config,
                                stopping_criteria=StoppingCriteriaList([stop])
                                )
        
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message  = ""
        for new_token in streamer:
            if new_token != '<':
                partial_message += new_token
                yield partial_message 
