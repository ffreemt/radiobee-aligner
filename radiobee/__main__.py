import gradio as gr

def greet(input):
    """Greet yo."""
    return f"'Sup yo! (your input: {input})"


if __name__ == "__main__":
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch()
