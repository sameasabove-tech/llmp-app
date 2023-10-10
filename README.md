# SSA Large Language Model Powered Application (llmp-app)


# API server 

The architecture has 3 main components:
- A hugging face llm backend
- A Gradio-based application layer mounted on an API endpoint
- A FastAPI backend, that loads the LLM model and provides endpoints as well as an interface to interact with it.



## Functionalities

1. Communication with the ModelClass language model: The script initializes an instance of the ModelClass class, which loads the ModelClass-13b model for language generation. It allows users to ask questions to the LLM and retrieve responses.
2. Dialog Management: The script manages conversations using the LlamaDialog class. It handles dialog history, system prompts, user questions, and assistant replies to maintain context in the conversation.
3. FastAPI Endpoints: The script defines several endpoints using FastAPI to interact with the LLM. These endpoints allow users to ask questions, display dialog history, clear the conversation history, and show the content of specific dialogs.
4. Data Models: The script includes the LlmCall data models for handling requests and responses to/from the LLM API.
5. Chat interface: the script also serves a chat interface built on Gradio, to interact easily with the LLM 


## Endpoints

- POST "/ask": Ask a question to the LLM. Provide the question in the request body. Returns the LLM response and the UUID of the dialog.
- GET "/ask": Provides a message instructing to use POST for asking questions.
- GET "/clear_history": Clears the conversation history and frees up memory. Returns a message indicating the history is cleared and the number of dialogs removed.
- GET "/delete_dialog": Deletes a specific dialog using its UUID.
- GET "/show_dialog": Shows the content of a specific dialog. Provide the UUID of the dialog in the request parameters.
- GET "/show_history": Shows the conversation history (dialogs).
- GET "/": Root endpoint showing the model name of the LLM (ModelClass).


## Project Directory Structure 
```
root
├── README.md
├── api_server.py
├── api_server_test_loic.py
└── src
    ├── backend
    │   ├── llm.py
    │   ├── llm_call.py
    │   └── llm_dialog.py
    └── frontend
        └── gradio_chat_interface.py
```

## Running the LLM Server

To start the LLM server, follow these steps:

1. Clone repo and navigate to root:

```shell
    git clone https://github.com/sameasabove-tech/llmp-app.git
    cd llmp-app
```

3. Create and activate virtual environment:
```shell
    python -m venv <env-path>
    source <env-path>/bin/activate
```
4. Run the server:
```shell
    pip install -r requirements.txt
```
5. Run the server:
```shell
    python api_server.py
```

Now, the LLM server should be up and running, and you can use the defined endpoints to interact with the SSA LLM API.

...

## Chat Web Interface

The LLM can be accessed directly with a chat interface that alows direct promtping and setting the main generation parameters.

You can access the chatbot at the following address:
```
https://...
```
_Note: The API server does not have a proper SSL certificate, so SSL warnings might appear, just ignore them._