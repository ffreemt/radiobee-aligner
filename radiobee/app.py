"""Talk to spaces VM via subprocess.check_output."""
# import httpx
import subprocess as sp
from shlex import split
import gradio as gr


def greet(command):
    """Probe vm."""
    try:
        out = sp.check_output(split(command), encoding="utf8")
    except Exception as e:
        out = str(e)
    # return "Hello " + name + "!!"
    if not (out and out.strip()):
        out = "No output, that's all we know."
    return out


iface = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    examples=[
        "cat /proc/version",
        "free  # show free memory",
        "uname -m",
        "df -h .",
        "cat /proc/cpuinfo",
    ],
    title="probe the system",
    description="talk to the system via subprocess.check_output ",
)

# iface.launch(share=True, debug=True)
iface.launch(debug=True)
