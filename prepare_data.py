import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_samsum', type=str, help='path to directory containing SAMSum corpus (train.json, test.json and val.json)')
parser.add_argument('--path_samsum_bart', type=str, help='path to samsum directory in bart directory')
parser.add_argument('--anonymise' , type=bool, default='False', help='whether to anonymise names or not')
parser.parse_args()

def get_all_speakers(dialogue): 
    utterances = dialogue.split('\n')
    return set(utterance[:utterance.index(':')] for utterance in utterances if ':' in utterance)

modes = ['train', 'test', 'val']
path_to_samsum = parser.path_samsum
output_dir = paser.path_samsum_bart

anonymise = parser.anonymise

for mode in modes:
    file_name = mode+'.json'
    with open(os.path.join(path_to_samsum, file_name), 'r') as f:
        json_content = json.load(f)
        f.close()
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, mode+'.source'), 'w+') as source_file, open(os.path.join(output_dir, mode+'.target'), 'w+') as target_file:
        for example in json_content:
            dialogue = example['dialogue'].lower()
            summary = example['summary'].lower()
            if anonymise:
                speakers = get_all_speakers(dialogue)
                speakers = dict(zip(speakers, range(len(speakers))))
                for speaker in speakers.keys():
                    dialogue = dialogue.replace(speaker, 'speaker{}'.format(speakers[speaker]))
                    summary = summary.replace(speaker, 'speaker{}'.format(speakers[speaker]))
            dialogue = dialogue.replace('\r\n', ' <sep> ')
            dialogue = dialogue.replace('\r', ' <sep> ')
            dialogue = dialogue.replace('\n', ' <sep> ')
            summary = summary.replace('\r\n', ' <sep> ')
            summary = summary.replace('\r', ' <sep> ')
            summary = summary.replace('\n', ' <sep> ')
            source_file.write(dialogue+'\n')
            target_file.write(summary+'\n')
        source_file.close()
        target_file.close() 
