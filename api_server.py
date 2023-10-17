
import os
import getpass

from fastapi import FastAPI
from torch.cuda import empty_cache
import uvicorn

from src.backend.llm_call import LLMCall
from src.backend.llm import ModelClass
from src.frontend.gradio_chat_interface import create_chat_interface
from src.backend.summarizer import SummarizerClass
from src.frontend.gradio_document_input_interface import create_input_document_interface
from src.backend.storage.database_connections import DocumentDatabase

import gradio as gr

'''
#test stream

llm = ModelClass()
generator = llm.ask_llm_stream(question="What is the largest country in the world?")
for response in generator:
    print(response)
'''              
         


"""
SAA-Tech LLM - Same As Above Large Language Model

This Python script provides an API for interacting with the SAA-Tech LLM (Same As Above Large Language Model). The API is built using FastAPI, a modern web framework for building APIs with Python.

The script contains the following functionalities:

1. Communication with the ModelClass language model: The script initializes an instance of the ModelClass class, which loads the ModelClass-13b model for language generation. It allows users to ask questions to the LLM and retrieve responses.
2. FastAPI Endpoints: The script defines several endpoints using FastAPI to interact with the LLM. These endpoints allow users to ask questions, display dialog history, clear the conversation history, and show the content of specific dialogs.

Usage:
1. Start the API by running this script.
2. Access the API endpoints using HTTP requests (GET/POST).

Endpoints:
- POST "/ask": Ask a question to the LLM. Provide the question in the request body. Returns the LLM response and the UUID of the dialog.
- GET "/ask": Provides a message instructing to use POST for asking questions.
- GET "/clear_history": Clears the conversation history and frees up memory. Returns a message indicating the history is cleared and the number of dialogs removed.
- GET "/delete_dialog": Deletes a specific dialog using its UUID.
- GET "/show_dialog": Shows the content of a specific dialog. Provide the UUID of the dialog in the request parameters.
- GET "/show_history": Shows the conversation history (dialogs).
- GET "/": Root endpoint showing the model name of the LLM (ModelClass).

Required Libraries:
- accelerate
- typing
- fastapi
- gradio
- torch
- transformers
- pathlib
- pydantic

Local script: 
- ModelClass.py
- llm_call.py
- gradio_chat_interface.py

Ensure that you have these libraries installed before running the script. The ModelClass model will be loaded during initialization.

Note:
- The script assumes the presence of the ModelClass class from the "ModelClass" module. 
Ensure that you have the "ModelClass.py" file with the "ModelClass" and "LlamaDialog" classes in the same directory as this script.

For further details on the usage of specific endpoints and classes, please refer to the comments and docstrings within the script.

"""
#Starting up the FastAPI
print("Setting up the FastAPI app...")
app = FastAPI()
# List of all the dialogs recieved by the API server
dialogs = []
# Loading up the ModelClass model
llm = ModelClass()
summarizer = SummarizerClass()
# Loading up the DocumentDatabase db
summarizer_db = DocumentDatabase("summarizer")

@app.get("/")
def read_root() ->str:
    """
    Root endpoint to show the model name.

    Returns:
        str: The model name of the LLM (ModelClass).
    """
    return f'SAA-Tech LLM : {llm.model_name}, {summarizer.model_name}' 

@app.post("/ask")
async def read_question(llm_call: LLMCall) -> dict:
    """
    Endpoint to receive a question and get the LLM response.

    Parameters:
        llm_call (LLMCall): The request containing the question and other parameters.

    Returns:
        dict: A dictionary containing the LLM response (message) and the UUID of the dialog.
    """
    global dialogs
    llm_response, uuid, warning_messages, debug_info = llm_call.ask_llm(llm=llm, dialogs=dialogs) 
   
    return {"message": llm_response, 'uuid': uuid, 'warnings': warning_messages, 'debug_info': debug_info}

@app.get("/ask")
async def get_ask() -> str:
    """
    Endpoint to provide a response for a GET request. (Returns a message instructing to use POST)
    """
    return 'SAA-Tech LLM , use POST'

@app.get("/clear_history")
async def clear_history() -> dict:
    """
    Endpoint to clear the conversation history. Deletes all dialogs in memory. 

    Returns:
        dict: A dictionary containing a message indicating that the history is cleared, and the number of dialogs removed.
    """
    global dialogs
    dialogs.clear()
    empty_cache()  # Freeing some of the memory
    return {"message": 'History cleared, all dialogs removed', 'history': len(dialogs)}

@app.get("/delete_dialog")
async def delete_dialog(uuid: str) -> str:
    """
    Endpoint to delete a specific dialog.

    Parameters:
        uuid (str): The UUID of the dialog to delete.

    Returns:
        str: Contains a string with the outcome.
    """
    global dialogs
    len_dialogs = len(dialogs)
    if uuid is None:
        raise(ValueError('Dialog uuid must be provided'))
    else:
        dialogs_without_uuid = [ditem for ditem in dialogs if ditem.UUID != uuid]
        dialogs.clear()
        dialogs.extend(dialogs_without_uuid) # This is needed because we are working with a global variable
        if len(dialogs)<len_dialogs:
            return (f"Dialog {uuid} removed!")
        else:
            return(f'Dialog {uuid} not found')

@app.get("/show_dialog")
async def show_dialog(uuid: str) -> dict:
    """
    Endpoint to show the content of a specific dialog.

    Parameters:
        uuid (str): The UUID of the dialog to display.

    Returns:
        Dict: Contains a string with the dialog content.
    """
    global dialogs
    if uuid is None:
        raise(ValueError('Dialog uuid must be provided'))
    else:
        dialog = [ditem for ditem in dialogs if ditem.UUID == uuid]
    if dialog:
        dialog = dialog[0] if isinstance(dialog, list) else dialog
    else:
        raise(Exception("Dialog not found"))

    return {'Dialog': dialog.display_dialog()}

@app.get("/show_history")
async def show_history() -> dict:
    """
    Endpoint to show the conversation history.

    Returns:
        dict: A dictionary containing the conversation history (dialogs).
    """
    global dialogs
    return {'history': dialogs}

#Gradio app for providing an interactive chat interface

interface_chat = create_chat_interface(delete_dialog, llm, dialogs)
interface_chat.queue(concurrency_count=40)
CHAT_PATH = '/chat'
app = gr.mount_gradio_app(app, interface_chat, path=CHAT_PATH)

interface_summarizer = create_input_document_interface(summarizer, summarizer_db)
interface_summarizer.queue(concurrency_count=40)
SUMMARIZER_PATH = '/summarize'
app = gr.mount_gradio_app(app, interface_summarizer, path=SUMMARIZER_PATH)

if __name__ == "__main__":
    print("Starting the API Server ...")
    uvicorn.run(app,
                host="0.0.0.0",
                port=5555
                )

              