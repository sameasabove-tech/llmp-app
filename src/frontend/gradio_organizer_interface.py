import json
import gradio as gr
from gradio import Blocks
import pandas as pd

from src.backend.organizer import OrganizerClass
from src.backend.storage.database_connections import DocumentDatabase
from src.backend.utlities.util_fns import uuid_factory, NpEncoder

'''Gradio app for providing a textbox input interface '''


CSS : str ="""
#chatbot { flex-grow: 1; overflow: auto; min-height: 0}
footer {visibility: hidden}
"""

def create_organizer_interface(llm: OrganizerClass, db: DocumentDatabase) -> Blocks:
    """
    Create a input interface to summarize/text embed a document with an LLM model/encoder using Gradio Blocks.

    Parameters:
        llm (SummarizerClass): the llm model to call
        db (DocumentDatabase): the global document database

    Returns:
            Blocks:  The input document interface
    """
    with gr.Blocks(css=CSS, title='SAA-Tech LLM') as interface:
        
        # testing workflow: 
        # 1) user clicks button, --> done
        # 2) utlis checks db for uuid with key provided & creates a unique one for USER --> done 
        # 3) csv is imported --> done
        # 4) model is called, uuid with key checks db for uuid and creates a unique for new document --> done
        # 5) document is embedded with uuid --> done 
        # 6) document embbedding with uuid is stored in db with labels

        dummy_data = pd.read_csv(f'src/backend/storage/david-test_data_subtopics.csv')

        def upload_file(files):
            file_paths = [file.name for file in files]
            return file_paths

        def call_llm_and_db(): #docs
            docs = list(dummy_data.text[6:106])
            user_id = uuid_factory()
            encodings = llm.encode_document(docs)
            for i in range(len(docs)):
                new_db_entry = {'user_id': user_id 
                            , 'doc_id': encodings[i][0]
                            , 'doc': docs[i]
                            , 'encoding': encodings[i][1].tolist()
                            , 'file_dir': None} #json.dumps(encodings[i][1], cls=NpEncoder)} # 'topic': ,'subtopic':} --> need labels here 
                # print(new_db_entry)
                db.write_to_database(new_db_entry)
            return 'Docs embedded!'
            # return None #'Docs embedded!'

        gr.Markdown("# SAA-Tech LLM")
        gr.Markdown("**This is a beta**")
        gr.Markdown("[Github](https://github.com/sameasabove-tech)")
        file_output = gr.File()
        upload_button = gr.UploadButton("Click to Upload a File", file_types=["text", "image", "video", "audio"], file_count="multiple")
        upload_button.upload(upload_file, upload_button, file_output)
        user_id = uuid_factory()
        btn = gr.Button(value="Run Text Emedding")
        txt = ["xxx"] # documents from imports 
        txt_3 = gr.Textbox(value="", label="Output")
        # txt_3 = "Documents Embedded"
        # for doc in txt:
        
        btn.click(call_llm_and_db, outputs=[txt_3]) #, outputs=[txt_3]) #, inputs=list(dummy_data.text[0:10]), outputs=[txt_3])

    return interface

   
