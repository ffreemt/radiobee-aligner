"""Convert file to text."""
from radiobee.process_upload import process_upload


def file2text(file1):
    """Convert file to text."""
    try:
        filename1 = file1.name
    except AttributeError:
        filename1 = ""
    if filename1:
        text1 = process_upload(file1)
    else:
        text1 = ""
    return text1
