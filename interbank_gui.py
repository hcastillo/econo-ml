from gooey import Gooey, GooeyParser
import sys


def graphical():
    try:
        import tkinter as tk
        tk.Tk().destroy()
    except:
        return False
    else:
        return True

@Gooey
def create_gooey():
    return gooey.GooeyParser(description="Interbank model")

def get_interactive_parser():
    if not sys.argv[1:] and graphical():
        return create_gooey()
    else:
        return None
