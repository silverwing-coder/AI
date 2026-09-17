
"""
$ pip isntall gradio
"""

import gradio as gd

def greet(name):
    return "Hello " + name + "!"

# demo = gd.Interface(fn=greet, inputs="text", outputs="text")
# demo = gd.Interface(fn=greet, inputs=gd.Textbox(lines=2, placeholder="Name here"), outputs="text")
demo = gd.Interface(fn=greet, inputs=["text", "checkbox", gd.Slider(0, 100)], outputs="text")
demo.launch()