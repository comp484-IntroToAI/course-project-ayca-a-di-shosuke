import tkinter as tk
from tkinter import scrolledtext


def generateSummary():
    input_text = input_text_box.get("1.0", tk.END)
    # summary = "You will see our wonderful summary here."
    output_text_box.config(state="normal")
    output_text_box.insert(tk.END, input_text.lower())
    output_text_box.config(state="disabled")


window = tk.Tk()
window.title("REvolutionary NLP")
width = window.winfo_screenwidth()  # /2
height = window.winfo_screenheight()
window.geometry("%dx%d" % (width, height))
# window.rowconfigure(0, minsize=100)
# window.columnconfigure(1, minsize=250)


input_text_box = scrolledtext.ScrolledText(
    width=window.winfo_width(),
    )
input_text_box.grid(row=0, column=0)
# input_text_box.delete("1.0", tk.END)


btn_summary = tk.Button(
    text="Generate Summary",
    width=20,
    height=3,
    fg="black",
    command=generateSummary
)
btn_summary.grid(row=1, column=0)


output_text_box = scrolledtext.ScrolledText(
    width=window.winfo_width(),
    state="disabled",
)
output_text_box.grid(row=2, column=0)

window.mainloop()
