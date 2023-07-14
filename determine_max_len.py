import matplotlib.pyplot as plt
from transformers import T5Tokenizer

from lib_new import load_data

tokenizer = T5Tokenizer.from_pretrained('base5')

data = load_data()
seq_src, seq_dst = zip(*data)

fig, axs = plt.subplots(2)

len_list_src = list(map(len, tokenizer(seq_src).input_ids))
len_list_dst = list(map(len, tokenizer(seq_dst).input_ids))

max_val = max(len_list_src) + 50
axs[0].hist(len_list_src, bins=range(0, max_val, 50), edgecolor='black')
axs[0].set_title('Src Length Distribution')
axs[0].set_xlabel('Ranges (Every 50)')
axs[0].set_ylabel('Count')
axs[0].set_xticks(range(0, max_val, 50))
axs[0].grid(True)

max_val = max(len_list_dst) + 10
axs[1].hist(len_list_dst, bins=range(0, max_val, 10), edgecolor='black')
axs[1].set_title('Dst Length Distribution')
axs[1].set_xlabel('Ranges (Every 50)')
axs[1].set_ylabel('Count')
axs[1].set_xticks(range(0, max_val, 10))
axs[1].grid(True)

fig.tight_layout()
plt.savefig('len_list.png')
