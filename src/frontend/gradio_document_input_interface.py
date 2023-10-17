import gradio as gr
from gradio import Blocks
from src.backend.summarizer import SummarizerClass
from src.backend.storage.database_connections import DocumentDatabase

'''Gradio app for providing a textbox input interface '''


CSS : str ="""
#chatbot { flex-grow: 1; overflow: auto; min-height: 0}
footer {visibility: hidden}
"""

def create_input_document_interface(llm: SummarizerClass, db: DocumentDatabase) -> Blocks:
    """
    Create a input interface to summarize/text embed a document with an LLM model/encoder using Gradio Blocks.

    Parameters:
        llm (SummarizerClass): the llm model to call
        db (DocumentDatabase): the global document database

    Returns:
            Blocks:  The input document interface
    """
    with gr.Blocks(css=CSS, title='SAA-Tech LLM') as interface:
        
        def call_llm_and_db(doc):
                new_db_entry = llm.summarize_document(doc)
                db.write_to_database(new_db_entry)
                return new_db_entry['summary']

        # uuid_var = gr.State()
        gr.Markdown("# SAA-Tech LLM")
        gr.Markdown("**This is a beta**")
        gr.Markdown("[Github](https://github.com/sameasabove-tech)")
        txt = gr.Textbox(label="Input Document", lines=7)
        txt_3 = gr.Textbox(value="", label="Summary Output")
        btn = gr.Button(value="Summarize")
        btn.click(call_llm_and_db, inputs=[txt], outputs=[txt_3]) 
        
    return interface
   
