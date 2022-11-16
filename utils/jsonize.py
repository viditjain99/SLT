import json 
import os
from argparse import ArgumentParser


parser = ArgumentParser(description="jsonize")

parser.add_argument('--dataset', type=str, default='aslg')
parser.add_argument('--mode', nargs='*', default=["train", "dev", "test"])
parser.add_argument('--src', type=str, default='en' )
parser.add_argument('--tgt', type=str, default='gloss.asl' )

args = parser.parse_args()

data_dir = "./data"

lang_code_map = {
  "gloss.asl" : "gl",
  "en": "en"
}

for mode in args.mode:
  src_filename = "{ds}.{mode}.{src}".format(ds = args.dataset, mode=mode, src=args.src)

  tgt_filename = "{ds}.{mode}_processed.{src}".format(ds = args.dataset,mode=mode, src=args.tgt)

  src_path = os.path.join(data_dir, src_filename)
  tgt_path = os.path.join(data_dir, tgt_filename)

  src_lines = open(src_path, 'r',encoding='utf-8-sig').readlines()
  tgt_lines = open(tgt_path, 'r', encoding='utf-8-sig').readlines()

  assert len(src_lines)==len(tgt_lines)

  output_filename = os.path.join(data_dir, "{mode}_{args.dataset}.json".format(mode=mode))
  output_file = open(output_filename, 'w')

  # examples = []
  for i, line in enumerate(src_lines):
    example = {'translation': {}}
    example['translation'][lang_code_map[args.src]] = line.strip()
    example['translation'][lang_code_map[args.tgt]] = tgt_lines[i].strip()

    # examples.append(example)

    json.dump(example, output_file)
    output_file.write("\n")

  
  


