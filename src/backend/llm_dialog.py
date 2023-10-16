from datetime import datetime
from typing import Dict, List, Literal,  Generator, Union
import uuid
import warnings

from pydantic import BaseModel, Field, validator, computed_field



"""
This script provides functionalities for interacting with the ModelClass-13b language model (LLM) . 
It includes data models, classes, and utility functions to manage conversations from the LLM. 

Functions:
1. uuid_factory: Generates and returns a UUID (Universally Unique Identifier).

Classes:
1. DialogEvent: Data model for a single event in a dialog. It stores attributes like UUID, role (system, user, or assistant), content, dialog UUID, and creation datetime.
2. LlamaDialog: Data model for managing conversations with the LLM. It keeps track of dialog events, system prompts, user questions, and assistant replies. It includes methods for adding events, asking questions, and displaying the entire dialog.

"""


def uuid_factory() -> str:
    """
    Generates and returns a UUID (Universally Unique Identifier).

    Returns:
        str: A unique identifier in the form of a string.
    """
    return str(uuid.uuid1())


class DialogEvent(BaseModel):
    """
    Data model for a single event in a dialog.

    Attributes:
        UUID (str): Universally Unique Identifier for the event.
        role (Literal["system", "user", "assistant"]): Role of the event participant (system, user, or assistant).
        content (str): Content of the event.
        llama_dialog_uuid (str): Universally Unique Identifier for the dialog containing this event.
        creation_datetime (datetime): Creation datetime of the event (defaults to current datetime).
    """
    UUID: str = Field(default_factory=uuid_factory)
    role: Literal["system", "user", "assistant"]
    content: Union[str, Generator] = ''
    content_len: int = 0 
    llama_dialog_uuid: str
    creation_datetime: datetime = datetime.now()

    @validator('content_len', always=True)
    def compute_content_len(cls, v: int, values: dict) -> int:
        return len(values['content'])


class LlamaDialog(BaseModel):
    """
    Data model for the LlamaDialog, which manages conversations.

    Attributes:
        UUID (str): Universally Unique Identifier for the dialog.
        no_history (bool): Flag indicating whether to retain the conversation history (default: False).
        creation_datetime (datetime): Creation datetime of the dialog (defaults to current datetime).
        system_prompt (str): The system prompt for the conversation.
        dialog (List[DialogEvent]): List of events representing the conversation.
        bos_token (str): Beginning of sentence token.
        eos_token (str): End of sentence token.
        B_INST (str): Beginning tag for system instructions.
        E_INST (str): End tag for system instructions.
        B_SYS (str): Beginning tag for the system response.
        E_SYS (str): End tag for the system response.
    
    Properties:
        dialog_len (int): number of characters in the entire dialog
    
    Methods:
        __init__(self, **kwargs): Initialization method to set the system prompt as the first dialog event.
        get_llm_formated_dialog(self) -> str: Formats the dialog for LLM in the required format.
        supplement_system_prompt(self, extra_system_prompt: str) -> None: Supplements the system prompt with additional text.
        replace_system_prompt(self, system_prompt: str) -> None: Replaces the default system prompt
        display_dialog(self) -> str: Displays the entire dialog with each event and turn.
        add_dialog_event(self, role: str, content: str) -> None: Adds a new dialog event.
        user_ask(self, content: str) -> None: Allows the user to ask a question.
        assistant_reply(self, content: str) -> None: Allows the assistant to reply.
    """

    UUID: str = Field(default_factory=uuid_factory)
    no_history: bool = False
    creation_datetime: datetime = datetime.now()
    system_prompt : str = '''
    You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.
    '''
    dialog : List[DialogEvent] = []
    bos_token: str = "<s>"
    eos_token: str = "<\s>"
    B_INST: str  = "[INST]"
    E_INST: str  = "[/INST]"
    B_SYS: str = "<<SYS>>\n" 
    E_SYS: str = "\n<</SYS>>\n\n"
        

    def __init__(self, **kwargs):
        """
        Initializes a LlamaDialog object.

        Parameters:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dialog = [DialogEvent(role='system', llama_dialog_uuid=self.UUID, content=self.system_prompt)]

    
    @computed_field
    @property
    def dialog_len(self) -> int:
        dialog_len = 0 
        for event in self.dialog:
            dialog_len += event.content_len
        return dialog_len

    def get_llm_formated_dialog(self) -> str:
        """
        Formats the dialog in the required format by Llama 2.

        Returns:
            str: The formatted dialog for Llama 2.
        """
        output_string = ''

        if self.dialog[-1].role != 'user':
            raise Exception('Last event must be from the user')
        if self.dialog[0].role != 'system':
            raise Exception('First dialog event must be system')
        if self.dialog_len/4 > 2048:
            warnings.warn('Dialog is likely to exceed the context window of 2048')
            
        new_dialog = [{
            "role": self.dialog[1].role,
            "content": self.B_SYS
            + self.dialog[0].content
            + self.E_SYS
            + self.dialog[1].content,
        }] + [{'role': d_event.role, 'content': d_event.content} for d_event in self.dialog[2:]]
        ndlist = [self.bos_token + f"{self.B_INST} {(prompt['content']).strip()} {self.E_INST} {(answer['content']).strip()} " + self.eos_token
                  for prompt, answer in zip(new_dialog[::2], new_dialog[1::2])]
        ndlist.append(self.bos_token + f"{self.B_INST} {(new_dialog[-1]['content']).strip()} {self.E_INST}")
        return ''.join(ndlist)

    def supplement_system_prompt(self, extra_system_prompt: str) -> None:
        """
        Supplements the system prompt with additional text.

        Parameters:
            extra_system_prompt (str): Additional text to add to the system prompt.
        """
        print(f"Appending the following prompt: \n {extra_system_prompt}")
        self.dialog[0].content = self.system_prompt + extra_system_prompt

    def replace_system_prompt(self, system_prompt: str) -> None:
        """
        Replaces the system prompt with additional text.

        Parameters:
            system_prompt (str): Text to replace the system prompt.
        """
        print(f"Appending the following prompt: \n {system_prompt}")
        self.dialog[0].content = system_prompt

    def display_dialog(self) -> str:
        """
        Displays the entire dialog with each event and turn.

        Returns:
            str: A formatted string displaying the dialog.
        """
        sep = '-' * 50 + '\n '
        output_string = sep

        for i, dialog_event in enumerate(self.dialog):
            output_string += " " + dialog_event.role.upper() + ": " + dialog_event.content + " \n"
            output_string += f'---Turn:{i + 1}' + sep
        return output_string

    def add_dialog_event(self, role: str, content: str) -> None:
        """
        Adds a new dialog event.

        Parameters:
            role (str): Role of the event participant (system, user, or assistant).
            content (str): Content of the event.
        """
        dialog_event = DialogEvent(role=role, llama_dialog_uuid=self.UUID, content=content)
        self.dialog.append(dialog_event)

    def user_ask(self, content: str) -> None:
        """
        Allows the user to ask a question.

        Parameters:
            content (str): The user's question.
        """
        if self.no_history:
            self.dialog = self.dialog[0:1]
        self.add_dialog_event(role='user', content=content)

    def assistant_reply(self, content: str) -> None:
        """
        Allows the assistant to reply.

        Parameters:
            content (str): The assistant's reply.
        """
        self.add_dialog_event(role='assistant', content=content)
