import tkinter as tk
from tkinter import scrolledtext
from EvolutionaryModel import GAHelpers


ga = GAHelpers()
vocab = ga.read_list('vocab_dict')[10000]
weights = ga.read_list('avg_fitness_g10_p50_a0010_v10000')
dictionary = ga.update_weights(vocab, weights)


def generate_summary():
    threshold = 0.7
    input_text = input_text_box.get("1.0", tk.END)
    input_text = ga.filter_sentence(input_text, "sentence")
    summary_lower = ga.summarize(dictionary, input_text, threshold)
    while len(summary_lower) < 10 and threshold > 0:
        threshold -= 0.001
        summary_lower = ga.summarize(dictionary, input_text, threshold)
        print(threshold)
    if threshold <= 0:
        output_text = "Cannot generate summary"
    else:
        summary = [sent.capitalize() for sent in summary_lower]
        summary = " ".join(summary)
        summary = ga.filter_sentence(summary, "sentence")
        output_text = summary
    output_text_box.config(state="normal")
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
    )
input_text_box.grid(row=0, column=0)


btn_summary = tk.Button(
    text="Generate Summary",
    width=20,
    height=3,
    fg="black",
    command=generate_summary
)
btn_summary.grid(row=1, column=0)


output_text_box = scrolledtext.ScrolledText(
    # width=window.winfo_width(),
    width=(int)(width/8.1),
    state="disabled",
)
output_text_box.grid(row=2, column=0)

window.mainloop()
