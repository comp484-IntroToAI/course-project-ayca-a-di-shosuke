import tkinter as tk
from tkinter import scrolledtext
from EvolutionaryModel import GAHelpers


def generate_summary():
    input_text = input_text_box.get("1.0", tk.END)
    input_text = ga.filter_sentence(input_text)
    summary = ga.summarize(dictionary, input_text, threshold)
    # summary = "Ethics in Machine Learning and Other Domain-Specific AI Algorithms Imagine, in the near future, a bank using a machine learning algorithm to recommend mortgage applications for approval. Sec- 3 The Ethics of Artificial Intelligence ond, if the programmers had manually input what they considered a good move in each possible situation, the resulting system would not have been able to make stronger chess moves than its creators. This may seem like an unenviable position from the perspective of public relations, but it’s hard to see what other guarantee of ethical behavior would be possible for a general intelligence operating on unforeseen problems, across domains, with preferences over distant consequences. Similarly, the Principle of Ontogeny Non-Discrimination is consistent with the claim that the creators or owners of an AI system with moral status may have special duties to their artificial mind which they do not have to another artificial mind, even if the minds in question are qualitatively similar and have the same moral status. To illustrate why some of our moral norms need to be rethought in the context of AI reproduction, it will suffice to consider just one exotic property of AIs: their capacity for rapid reproduction. It may seem premature to speculate, but one does suspect that some AI paradigms are more likely than others to eventually prove conducive to the creation of intelligent self- modifying agents whose goals remain predictable even after multiple iterations of self- 15 The Ethics of Artificial Intelligence improvement."
    print(summary)
    output_text_box.config(state="normal")
    output_text_box.insert(tk.END, summary)
    output_text_box.config(state="disabled")


ga = GAHelpers()
vocab = ga.read_list('vocab_dict')[10000]
weights = ga.read_list('new_vocab')
dictionary = ga.update_weights(vocab, weights)
# dictionary = {'new': 0.9, 'test': 0.1} # TODO: replace with line above once model is trained
threshold = 0.05


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
    command=generate_summary
)
btn_summary.grid(row=1, column=0)


output_text_box = scrolledtext.ScrolledText(
    width=window.winfo_width(),
    state="disabled",
)
output_text_box.grid(row=2, column=0)

window.mainloop()
