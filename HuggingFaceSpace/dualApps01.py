"""
$ pip install huggingface_hub
$ hf auth login                 -> follow instructions 
"""

import gradio as ui

# Define App 1
with ui.Blocks() as app1:
    ui.Markdown("# 🤖 Text Generator App")
    # Add your first app's inputs, buttons, and logic here

# Define App 2
with ui.Blocks() as app2:
    ui.Markdown("# 🎨 Image Generator App")
    # Add your second app's inputs, buttons, and logic here

# Combine them into one unified interface
demo = ui.TabbedInterface([app1, app2], ["Text App", "Image App"])

if __name__ == "__main__":
    demo.launch()