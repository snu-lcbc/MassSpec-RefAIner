from parameters import *

import os, argparse
import sentencepiece as spm

train_frac = 0.960

# def train_sp(model_type, vocab_size, is_src=True):
def train_sp(vocab_size, is_src=True):
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --max_sentence_length=999999 \
                --model_type={}"


    if is_src:
        this_input_file = f"{DATA_DIR}/{SRC_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{src_model_prefix}"

        config = template.format(this_input_file,
                                pad_id,
                                sos_id,
                                eos_id,
                                unk_id,
                                this_model_prefix,
                                vocab_size,
                                character_coverage,
                                sp_model_type)

        print(config)

        if not os.path.isdir(SP_DIR):
            os.mkdir(SP_DIR)

        print(spm)
        spm.SentencePieceTrainer.Train(config)



    else:
        this_input_file = f"{DATA_DIR}/{TRG_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{trg_model_prefix}"

        config = template.format(this_input_file,
                                pad_id,
                                sos_id,
                                eos_id,
                                unk_id,
                                this_model_prefix,
                                vocab_size,
                                character_coverage,
                                sp_model_type)

        print(config)

        if not os.path.isdir(SP_DIR):
            os.mkdir(SP_DIR)

        print(spm)
        spm.SentencePieceTrainer.Train(config)


def split_data(augment=False, num_augment=5):
    if augment:
        print('*'*3, f'\nTrain set of SRC will be augmented {num_augment} times.')

    src_trg = []
    for src, trg in zip( open(f"{DATA_DIR}/{SRC_RAW_DATA_NAME}", encoding="utf-8"),\
                   open(f"{DATA_DIR}/{TRG_RAW_DATA_NAME}", encoding="utf-8")):
        src_trg.append([src.strip(), trg.strip()])

    print("\nSplitting data...")
    temp = src_trg#[:int(train_frac * len(src_trg))]

    train_lines = temp[:int(train_frac * len(temp))]
    valid_lines = temp[int(train_frac * len(temp)):]
    test_lines =  valid_lines #src_trg[int(train_frac * len(src_trg)):]
    print(f'{len(src_trg) = }\n {len(train_lines) = }\n {len(valid_lines) = }\n {len(test_lines) = }\n')

    if not os.path.isdir(f"{DATA_DIR}/{SRC_DIR}"):
        os.mkdir(f"{DATA_DIR}/{SRC_DIR}")

    if not os.path.isdir(f"{DATA_DIR}/{TRG_DIR}"):
        os.mkdir(f"{DATA_DIR}/{TRG_DIR}")

    with open(f"{DATA_DIR}/{SRC_DIR}/{TRAIN_NAME}", 'w', encoding="utf-8") as srcf,\
    open(f"{DATA_DIR}/{TRG_DIR}/{TRAIN_NAME}", 'w', encoding="utf-8") as trgf:
        for src, trg in train_lines:
            if augment:
                for _ in range(num_augment):
                    src_list = src.strip().split()
                    shuffle(src_list)
                    shuffled_list = ' '.join(src_list)
                    srcf.write(shuffled_list + '\n')
                    trgf.write(trg+'\n')
            else:
                srcf.write(src + '\n')
                trgf.write(trg+'\n')

    with open(f"{DATA_DIR}/{SRC_DIR}/{VALID_NAME}", 'w', encoding="utf-8") as srcf,\
    open(f"{DATA_DIR}/{TRG_DIR}/{VALID_NAME}", 'w', encoding="utf-8") as trgf:
        for src, trg in valid_lines:
            srcf.write(src + '\n')
            trgf.write(trg + '\n')

    with open(f"{DATA_DIR}/{SRC_DIR}/{TEST_NAME}", 'w', encoding="utf-8") as srcf,\
    open(f"{DATA_DIR}/{TRG_DIR}/{TEST_NAME}", 'w', encoding="utf-8") as trgf:
        for src, trg in test_lines:
            srcf.write(src + '\n')
            trgf.write(trg + '\n')

    print(f"Train/Valid/test data saved in {DATA_DIR}/{SRC_DIR} and {DATA_DIR}/{TRG_DIR}.\n")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_type', default='ae', type=str, help='"smiles", "fp1", "ae"')
    parser.add_argument('--augment', action='store_true', help='Data augmentation of SRC (Products)')
    parser.add_argument('--num_augment', type=int, default=5, help='Number of data augmentation of SRC (Products)')

    args = parser.parse_args()
    train_sp(src_vocab_size, is_src=True)
    train_sp(trg_vocab_size, is_src=False)
    split_data()
    # if args.model_type in ["atom", "fp1", "ae", 'spe', 'char', 'kmer']:
    #     train_sp(args.model_type, src_vocab_size[args.model_type], is_src=True)
    #     train_sp(args.model_type, trg_vocab_size[args.model_type], is_src=False)
    #     split_data(args.model_type)
    #
    # elif args.model_type == 'all':
    #     train_sp('ae', src_vocab_size['ae'], is_src=True)
    #     train_sp('ae', trg_vocab_size['ae'], is_src=False)
    #     split_data('ae')
    #
    #     train_sp('smiles', src_vocab_size['smiles'], is_src=True)
    #     train_sp('smiles', trg_vocab_size['smiles'], is_src=False)
    #     split_data('smiles')
    #
    #     train_sp('fp1', src_vocab_size['fp1'], is_src=True)
    #     train_sp('fp1', trg_vocab_size['fp1'], is_src=False)
    #     split_data('fp1')
    # else:
    #     print('Please select one of these option: "smiles", "fp1", "ae", or "all"')
