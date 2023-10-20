import json
import gradio as gr
from gradio import Blocks
import pandas as pd

# from src.backend.organizer import OrganizerClass
# from src.backend.storage.database_connections import DocumentDatabase
# from src.backend.utlities.util_fns import uuid_factory, NpEncoder

'''Gradio Directory to Apps'''


CSS : str ="""
#chatbot { flex-grow: 1; overflow: auto; min-height: 0}
footer {visibility: hidden}
"""

def create_main_page_interface(endpoint_dict: dict, host_url: str) -> Blocks:
    """
    Create a input interface to summarize/text embed a document with an LLM model/encoder using Gradio Blocks.

    Parameters:
        llm (SummarizerClass): the llm model to call
        db (DocumentDatabase): the global document database

    Returns:
            Blocks:  The input document interface
    """
    with gr.Blocks(css=CSS, title='SAA-Tech LLM x Natural Lingo') as interface:

        gr.Markdown("# SAA-Tech x Natural Lingo")
        gr.Markdown("**This is a beta**")
        gr.Markdown("[Github](https://github.com/sameasabove-tech)")

        # quick and dirty main page 
        gr.Button(value='Chatbot', link=host_url+endpoint_dict['chatbot'])
        gr.Button(value='Summarizer', link=host_url+endpoint_dict['summarizer'])
        gr.Button(value='Organizer', link=host_url+endpoint_dict['organizer'])
        
        
        # file_output = gr.File()
        # upload_button = gr.UploadButton("Click to Upload a File", file_types=["text", "image", "video", "audio"], file_count="multiple")
        # upload_button.upload(upload_file, upload_button, file_output)
        # user_id = uuid_factory()
        # btn = gr.Button(value="Run Text Emedding")
        # txt = ["xxx"] # documents from imports 
        # txt_3 = gr.Textbox(value="", label="Output")
        # # txt_3 = "Documents Embedded"
        # # for doc in txt:
        
        # btn.click(call_llm_and_db, outputs=[txt_3]) #, outputs=[txt_3]) #, inputs=list(dummy_data.text[0:10]), outputs=[txt_3])

    return interface

   
