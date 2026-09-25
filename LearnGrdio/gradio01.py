"""
Edited by Sangmork Park at VMI
Last update: Sep. 2026
"""

# print('gradio test')

# $ pip install gradio
import gradio as gd

def greet(name):
    return "Hello " + name + "!"

demo = gd.Interface(fn=greet, inputs="text", outputs="text")

# default port number of gradio app is 7860: "localhost:7860"
# change port number by "demo.launch(server_port=8000)", or "export GRADIO_SERVER_PORT=8000"
demo.launch(server_port=8000)