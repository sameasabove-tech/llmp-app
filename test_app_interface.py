import gradio as gr

def process_files(files):
    file_names = [f.name for f in files]

    for file in files:
        print(f"FILE:: {file}")
        print(f"FILE NAME:: {file.name}")

    #for file in files:
    #    with open('readme.txt') as f:
    #        lines = f.readlines()

    return ",\n".join(file_names)

demo = gr.Interface(
    process_files,
    inputs='files',
    outputs="textbox"
)

demo.launch()