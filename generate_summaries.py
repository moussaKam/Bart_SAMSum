import torch
from fairseq.models.bart import BARTModel
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--checkpoint', type=str, help='checkpoint to generate summaries')
parser.add_argument('--summaries_file', type=str, help='file to save generated summaries')
parser.add_argument('--test_source', type=str, help='test dialogues')
parser.parse_args()

bart = BARTModel.from_pretrained(
    'checkpoints',
    checkpoint_file=args.checkpoint,
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open(args.test_source) as source, open(args.summaries_file, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=4)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=100, min_len=5, no_repeat_ngram_size=4)

        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
