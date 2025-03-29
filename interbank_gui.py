import sys
import gooey


def graphical():
    try:
        import tkinter as tk
        tk.Tk().destroy()
    except ModuleNotFoundError:
        return False
    else:
        return True


@gooey.Gooey
def create_gooey():
    return gooey.GooeyParser(description="Interbank model")


def get_interactive_parser():
    if not sys.argv[1:] and graphical():
        return create_gooey()
    else:
        return None
