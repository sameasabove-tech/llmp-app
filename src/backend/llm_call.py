import warnings
from src.backend.llm import ModelClass
from src.backend.llm_dialog import LlamaDialog
from pydantic import BaseModel
from typing import Dict, Union
from transformers import GenerationConfig

"""
Data model for making  LLMCalls

provides two methods:
ask_llm (ModelClass, list[LlamaDialogs]): gets a dialog, asks the question to the llm, and updates the conversation history
__get_dialog (dialogs): returns a dialog coresponding the the uuid from the LLMCall or creates a new one if not found 

"""


class LLMCall(BaseModel):
    """
    Data model for the request to the "/ask" endpoint.
    """
    question: str
    uuid: str = None
    no_history: bool = False
    debug: bool = False
    system_prompt: str = None
    generation_parameters: Union[Dict, GenerationConfig, None] = None

    class Config:
        arbitrary_types_allowed = True

    def __get_dialog(self, dialogs : list[LlamaDialog]) -> LlamaDialog:
        # Finding the correct dialog or creating one if this is a new conversation

        try:
            dialog = [ditem for ditem in dialogs if ditem.UUID == self.uuid][0]
        except IndexError:
            warnings.warn(f"Dialog uuid {self.uuid} not found. Creating a new dialog")
            dialog = LlamaDialog(no_history=self.no_history)
            dialogs.append(dialog)
            self.uuid = dialog.UUID
        return dialog
            
    def ask_llm(self, 
                llm: ModelClass, 
                dialogs: list[LlamaDialog]
                ) -> tuple:
        """
        Ask the LLM a question and handle the conversation history.

        Parameters:
            llm (ModelClass): ModelClass model to prompt
            dialogs (list[LlamaDialog]): List of all dialogs

        Returns:
            tuple: A tuple containing the LLM response (str), the UUID (str) of the dialog, any warning messages to pass onto the API caller and debug information if requested.
        """
        print('\n\n\n\nHI LOIC\n\n\n\n\n')
        dialog = self.__get_dialog(dialogs)
        # Adding to the system prompt
        if self.system_prompt:
            dialog.supplement_system_prompt(extra_system_prompt=self.system_prompt)
        # Expanding the conversation
        dialog.user_ask(self.question)
        formated_dialog = dialog.get_llm_formated_dialog() # reformats the question to specific LLama 2 format 
        result, warning_messages, debug_info = llm.ask_llm(formated_dialog, debug=self.debug, **self.generation_parameters)
        dialog.assistant_reply(result)
        return result, self.uuid, warning_messages, debug_info
