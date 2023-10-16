import gradio as gr
from gradio import Blocks

from src.backend.llm import ModelClass
from src.backend.llm_dialog import LlamaDialog

from typing import Tuple, Callable, List


'''Gradio app for providing an interactive chat interface '''


CSS : str ="""
#chatbot { flex-grow: 1; overflow: auto; min-height: 0}
footer {visibility: hidden}
"""

def create_chat_interface(delete_dialog: Callable, llm: ModelClass, dialogs: List[LlamaDialog] ) -> Blocks:
    """
    Create a chat interface with the LLM model using the Gradio Blocks

    Parameters:
            delete_dialog (callable): function to delete a dialog, used by the clear button in the interface.
            llm (ModelClass): the llm model to call
            dialogs (List[LlamaDialog]): global list of all the active dialogs 

    Returns:
            Blocks:  The chat interface
    
    """
   

    with gr.Blocks(css=CSS, title='SAA-Tech LLM') as interface:
        uuid_var = gr.State()
        gr.Markdown("# SAA-Tech LLM")
        gr.Markdown("**This is a beta**")
        gr.Markdown("[Github](https://github.com/sameasabove-tech)")
        with gr.Row():
            with gr.Column(variant="panel", scale=4, min_width=500):
                chatbot = gr.Chatbot(label="Chat", elem_id="chatbot", height=500)
                with gr.Group():
                    with gr.Row():
                            textbox = gr.Textbox(
                                    container=False,
                                    show_label=False,
                                    placeholder="Type a message...",
                                    scale=7,
                                    autofocus=True,
                                )
                            submit_btn = gr.Button(
                                        "Submit",
                                        variant="primary",
                                        scale=1,
                                        min_width=150,
                                    )
                    with gr.Row():
                        clear = gr.ClearButton(components=[chatbot, textbox, uuid_var])
            with gr.Column(variant="panel", scale=1):
                parameters_tile = gr.Markdown("## Parameters")
                temperature_slider  = gr.Slider(value=0.5, label="Temperature", minimum=0.01, maximum=2.0, step=0.1)
                top_p = gr.Slider(value=0.95, label="Top P", minimum=0.0, maximum=1.0)
                top_k  = gr.Slider(label="Top k", minimum=1, maximum=2000, value=100, step=1)
                max_new_tokens = gr.Slider(value=128, label="Max new tokens", minimum=0.0, maximum=2048, step=1)
                gr.Markdown('[Parameters explanation](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/)')
                #num_beams = gr.Slider(value=1, label="Num beams", minimum=1, maximum=4, step=1)  # Not available with streaming
                with gr.Box():
                    system_prompt_tile = gr.Markdown("### System prompt")
                    system_prompt = gr.Textbox(label=None, container=True, show_label=False, lines=7, max_lines=50, elem_id="system_prompt_textbox")
                    system_prompt_radio = gr.Radio(choices=['Replace', 'Extend'], value='Extend', container=False)
    
        def user(user_message, history):
            return "", history + [[user_message, None]]

        def __get_dialog(uu_id : str = None) -> LlamaDialog:
            
            if uu_id:
                dialog = [ditem for ditem in dialogs if ditem.UUID == uu_id]
                dialog = dialog[0] if isinstance(dialog, list) else dialog
            else:
                dialog = LlamaDialog()
            return dialog

        def get_system_prompt(uu_id: str = None, system_prompt_radio: str = 'Extend') -> Tuple[str, str]:
            
            if system_prompt_radio == 'Replace' :
                dialog = __get_dialog(uu_id)
                if uu_id is None:
                    dialogs.append(dialog)
                return dialog.dialog[0].content.strip(), dialog.UUID
            else:
                return "", uu_id
        
        def respond_streaming(chat_history, uu_id, temperature, top_p, top_k, max_new_tokens, system_prompt_radio, system_prompt):
            """
            When submitting the textbox adds the user's question to the dialog history and then submits the prompt to the llm.
            Expects to recieve a generator and then populates the chat history box with the streaming response
            Once the streaming is over, updates the dialog history with the bot's response 

            """
            dialog = __get_dialog(uu_id)

            prompt_fn = dialog.replace_system_prompt if system_prompt_radio == 'Replace' else dialog.supplement_system_prompt

            if system_prompt:
                prompt_fn(system_prompt)

            dialog.user_ask(chat_history[-1][0])
            generator = llm.ask_llm_stream(dialog.get_llm_formated_dialog(), temperature=temperature, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
            chat_history[-1][1] = ""
            for response in generator:
                chat_history[-1][1] = response
                yield(chat_history, dialog.UUID)
            dialog.assistant_reply(str(chat_history[-1][1]))
            dialogs.append(dialog)


        dict_streaming_predict = dict(fn=respond_streaming,
                                    inputs=[chatbot, uuid_var, temperature_slider, top_p, top_k, max_new_tokens, system_prompt_radio, system_prompt],
                                    outputs=[chatbot, uuid_var],
                                    show_progress=True)

        dict_transfer_input = dict(fn=user,
                                inputs=[textbox, chatbot],
                                outputs=[textbox, chatbot],
                                queue=False,
                                show_progress=True)                                    
        
        textbox.submit(**dict_transfer_input).then(**dict_streaming_predict)
        submit_btn.click(**dict_transfer_input).then(**dict_streaming_predict)
        clear.click(delete_dialog, [uuid_var], []).then(lambda: (None, 'Extend', '') , [],[uuid_var,  system_prompt_radio, system_prompt])
        system_prompt_radio.input(get_system_prompt, inputs=[uuid_var, system_prompt_radio], outputs=[system_prompt, uuid_var])
    interface.ssl_verify = False
    interface.show_api = False
    return interface

