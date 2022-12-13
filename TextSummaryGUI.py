import tkinter as tk
from tkinter import scrolledtext, Entry, Label
from EvolutionaryModel import GAHelpers


ga = GAHelpers()
vocab = ga.read_list('vocab_dict')[1000]
weights = ga.read_list('./vocab_files/new_vocab_g10_p50_a0100_v01000')
dictionary = ga.update_weights(vocab, weights)


def get_output_text(input_text):
    threshold = (float) (input_threshold_box.get())
    summary_lower = ga.summarize(dictionary, input_text, threshold)
    while len(summary_lower) < 10 and threshold > 0:
        threshold -= 0.001
        summary_lower = ga.summarize(dictionary, input_text, threshold)
    if threshold != (float) (input_threshold_box.get()):
        input_threshold_box.delete(0, tk.END)
        input_threshold_box.insert(0, round(threshold, 3))
    if threshold <= 0:
        return "Cannot generate summary"
    else:
        summary = [sent.capitalize() for sent in summary_lower]
        summary = " ".join(summary)
        summary = ga.filter_sentence(summary, "sentence")
        return summary


def generate_summary():
    input_text = input_text_box.get("1.0", tk.END)
    input_text = ga.filter_sentence(input_text, "sentence")
    output_text = get_output_text(input_text)
    output_text_box.config(state="normal")
    output_text_box.delete("1.0", tk.END)
    output_text_box.insert(tk.END, output_text)
    output_text_box.config(state="disabled")


window = tk.Tk()
window.title("REvolutionary NLP")
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
window.geometry("%dx%d" % (width, height))
window.update_idletasks()


input_text_box = scrolledtext.ScrolledText(
    # width=window.winfo_width(),
    width=(int)(width/8.1),
    # height=(int)(height/80)
    )
input_text_box.grid(row=0, column=0)
# input_text_box.pack(side="top")


btn_summary = tk.Button(
    text="Generate Summary",
    width=20,
    height=3,
    fg="black",
    command=generate_summary
)
# btn_summary.pack(side="top", pady=10)
btn_summary.grid(row=1, column=0, pady=10)



output_text_box = scrolledtext.ScrolledText(
    # width=window.winfo_width(),
    width=(int)(width/8.1),
    # height=(int)(height/80),
    state="disabled",
)
# output_text_box.pack(side="top")
output_text_box.grid(row=2, column=0)



footer_frame = tk.Frame(window)
input_threshold_label = Label(footer_frame, text = '  Threshold ', font=('calibre', 10, 'bold'))
input_threshold_box = Entry(footer_frame, font=('calibre', 10,'normal'))
input_threshold_box.insert(0, "0.03")
input_threshold_desc = Label(footer_frame, text = '   *Decrease threshold if "Cannot generate summary" or for a longer summary, and increase threshold for a shorter summary. Threshold decreases automatically if too high to generate a summary.', font=('calibre', 10, 'normal'))

# footer_frame.pack(side="top", anchor="nw", pady=10)
footer_frame.grid(row=3, column=0, pady=10)
input_threshold_label.pack(side="left")
input_threshold_box.pack(side="left")
input_threshold_desc.pack(side="left")


window.mainloop()
