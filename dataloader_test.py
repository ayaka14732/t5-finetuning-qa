from lib.proc_init_utils import initialise_cpu, initialise_tpu
initialise_cpu()


from transformers import T5Tokenizer

from lib.proc_init_utils import initialise_cpu, initialise_tpu
from lib_new import MyDataLoader, load_data

initialise_tpu('v4-16', n_devices=1, rank=3)
initialise_cpu()
def main() -> None:
    batch_size = 2
    max_len_enc = 512
    max_len_dec = 64


    tokenizer = T5Tokenizer.from_pretrained('base5')

    data = load_data()
    dataloader = MyDataLoader(data, tokenizer, batch_size, max_len_enc, max_len_dec)  # TODO: prng

    for src, dst, src_mask, dst_mask, labels in dataloader:
        print(src, dst, src_mask, dst_mask, labels, sep='\n\n')
        print(src.device())
        exit(-1)

if __name__ == '__main__':
    main()
