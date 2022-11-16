import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, concatenate_datasets, interleave_datasets

import evaluate
from responses import target
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.models.mbart.modeling_mbart import MBartModel, MBartForConditionalGeneration
from transformers import MBartConfig, AutoConfig, AutoModelForSeq2SeqLM

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# %%
os.system("python utils/jsonize.py --dataset aslg --mode train dev test --src en --tgt gloss.asl")
os.system("python utils/jsonize.py --dataset ncslgr --mode train dev test --src en --tgt gloss.asl")

# %%
# f1 = open("./data/aslg.train.en", 'r', encoding = 'utf-8-sig').readlines()
# f2 = open("./data/aslg.train_processed.gloss.asl", 'r', encoding = 'utf-8-sig').readlines()

# with open("./data/text_dump.txt", 'w') as f:
#   f.writelines(f1+f2)
#   f.close()

# %%

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    src_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: str = field(default=None, metadata={"help": "Target language id for translation."})
    vocab_size: int = field(default=None, metadata={"help": "vocab size for training custom tokenizer"})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    ood_train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    ood_validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    ood_test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )

    id_train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    id_validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    id_test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )

    train_stage: str = field(default=None, metadata={"help": "Source language id for translation."})

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.src_lang is None or self.tgt_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))


ood_dataset = "aslg"
id_dataset = "ncslgr"

# %%

model_args, data_args, training_args = parser.parse_args_into_dataclasses([
    "--train_stage", "mixed",  # ood/mixed/id
    "--ood_train_file", f"./data/train_{ood_dataset}.json",
    "--ood_validation_file", f"./data/dev_{ood_dataset}.json",
    "--ood_test_file", f"./data/test_{ood_dataset}.json",
    "--id_train_file", f"./data/train_{id_dataset}.json",
    "--id_validation_file", f"./data/dev_{id_dataset}.json",
    "--id_test_file", f"./data/test_{id_dataset}.json",
    "--model_name_or_path", "none", 
    "--output_dir" , "/results",
    "--src_lang", "gl_EN",
    "--tgt_lang", "en_XX",
    "--vocab_size", "6000",
    "--max_source_length", "50",
    "--max_target_length", "50",
    "--ignore_pad_token_for_loss", "True",
    "--output_dir","./result",
    "--generation_max_length", "50",
    "--generation_num_beams", "1",
    "--predict_with_generate", "True",
    "--per_device_train_batch_size", "16",
    "--per_device_eval_batch_size", "16",
    "--num_train_epochs", "50",
    "--learning_rate", "1e-5",
    "--do_train", "True",
    "--do_eval", "True",
    "--do_predict", "True",
    "--save_strategy", "no",
    "--logging_steps", "100",
    "--report_to", "wandb",
    ])

# %%
extension = "json"
ood_train_dataset = load_dataset( extension, data_files = data_args.ood_train_file, split = "train" )
ood_validation_dataset = load_dataset( extension, data_files = data_args.ood_validation_file, split = "train" )
ood_test_dataset = load_dataset( extension, data_files = data_args.ood_test_file, split = "train" )

id_train_dataset = load_dataset( extension, data_files = data_args.id_train_file, split = "train" )
id_validation_dataset = load_dataset( extension, data_files = data_args.id_validation_file, split = "train" )
id_test_dataset = load_dataset( extension, data_files = data_args.id_test_file, split = "train" )

total_train_dataset = concatenate_datasets([ood_train_dataset, id_train_dataset])
# %%
def get_training_corpus():
    return (
        [ p for k in total_train_dataset[i: i+1000]["translation"] for p in list(k.values()) ]
        for i in range(0, len(total_train_dataset), 1000)
    )
corpus = get_training_corpus()


def replace_gl(lang):
    return lang.replace("gl_EN", "ro_RO")

old_tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-en-ro", src_lang = replace_gl(data_args.src_lang), tgt_lang = replace_gl(data_args.tgt_lang))

tokenizer = old_tokenizer.train_new_from_iterator(corpus, data_args.vocab_size, new_special_tokens = ["gl_EN", "<id>", "<ood>"])
tokenizer.tgt_lang = data_args.tgt_lang
tokenizer.src_lang = data_args.src_lang
tokenizer.lang_code_to_id["gl_EN"] = tokenizer.convert_tokens_to_ids("gl_EN")
print("-"*100 + "\n")
print(f"Trained the tokenizer successfully with a vocab size: {len(tokenizer)}")

src_text = total_train_dataset[0]["translation"][data_args.src_lang]
print(f"Example source language text: {src_text}")
src_labels = tokenizer(text = src_text, max_length = 10, padding = "max_length", return_tensors="pt").input_ids
print(tokenizer.convert_ids_to_tokens(list(src_labels[0])))

tgt_text = total_train_dataset[0]["translation"][data_args.tgt_lang]
print(f"Example target language text: {src_text}")
tgt_labels = tokenizer(target_text = src_text, max_length = 10, padding = "max_length", return_tensors="pt").input_ids
print(tokenizer.convert_ids_to_tokens(list(tgt_labels[0])))
print("-"*100 + "\n")

padding = False
prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

source_lang = data_args.source_lang.split("_")[0]
target_lang = data_args.target_lang.split("_")[0]

def preprocess_function(examples, domain_tag = None):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    if domain_tag:
        for i in range(len(model_inputs["input_ids"])):
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:-1] + tokenizer.convert_tokens_to_ids(domain_tag) + [model_inputs["input_ids"][i][-1]]
            model_inputs["attention_mask"][i].append(1)

    # Tokenize targets with the `text_target` keyword argument
    # labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

    # adding domain tag is required for target??
    # for i in range(len(labels["input_ids"])):
    #     labels["input_ids"][i] = labels["input_ids"][i][:-1] + tokenizer.convert_tokens_to_ids(["<ood>"]) + [labels["input_ids"][i][-1]]

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

column_names = ood_train_dataset.column_names

print("-"*100)
print("\n Preprocessing out of domain datasets")

ood_train_dataset = ood_train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                fn_kwargs= {"domain_tag": "<ood>"},
                desc="Running tokenizer on train dataset",
            )

ood_eval_dataset = ood_validation_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                fn_kwargs= {"domain_tag": "<ood>"},
                desc="Running tokenizer on eval dataset",
            )
ood_test_dataset = ood_test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                fn_kwargs= {"domain_tag": "<ood>"},
                desc="Running tokenizer on test dataset",
            )

ids = list(ood_train_dataset[0]["input_ids"])
label_ids = list(ood_train_dataset[0]["labels"])
print(tokenizer.convert_ids_to_tokens(ids))
print(tokenizer.convert_ids_to_tokens(label_ids))
print(ood_train_dataset[0])

train_dataset = ood_train_dataset
eval_dataset = ood_eval_dataset
test_dataset = ood_test_dataset

if data_args.train_stage in {"id", "mixed"}:
    
    print("\n Preprocessing in domain datasets\n")
    
    id_train_dataset = id_train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs= {"domain_tag": "<id>"},
                    desc="Running tokenizer on train dataset",
                )

    id_eval_dataset = id_validation_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs= {"domain_tag": "<id>"},
                    desc="Running tokenizer on eval dataset",
                )
    id_test_dataset = id_test_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs= {"domain_tag": "<id>"},
                    desc="Running tokenizer on test dataset",
                )

    ids = list(id_train_dataset[0]["input_ids"])
    label_ids = list(id_train_dataset[0]["labels"])
    print(tokenizer.convert_ids_to_tokens(ids))
    print(tokenizer.convert_ids_to_tokens(label_ids))
    print(id_train_dataset[0])

    train_dataset = id_train_dataset
    eval_dataset = id_eval_dataset
    test_dataset = id_test_dataset

if data_args.train_stage == "mixed":
    
    print("\nCreating mixed dataset with oversampling\n")
    
    mixed_train_dataset = interleave_datasets(
            [ood_train_dataset, id_train_dataset], 
            probabilities = [0.2, 0.8], seed = training_args.seed)

    train_dataset = mixed_train_dataset

print("Finished dataset preprocessing")
print("-"*100)


# config = AutoConfig.from_pretrained(
#     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
#     cache_dir=model_args.cache_dir,
#     revision=model_args.model_revision,
#     use_auth_token=True if model_args.use_auth_token else None,
# )

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     model_args.model_name_or_path,
#     from_tf=bool(".ckpt" in model_args.model_name_or_path),
#     config=config,
#     cache_dir=model_args.cache_dir,
#     revision=model_args.model_revision,
#     use_auth_token=True if model_args.use_auth_token else None,
# )

config = MBartConfig()

# EMBEDDING_DIM = 512
# SCALE_DOWN_FACTOR = 4

# config.d_model = EMBEDDING_DIM
# config.vocab_size = data_args.vocab_size
# config.encoder_attention_heads //= SCALE_DOWN_FACTOR
# config.encoder_ffn_dim //= SCALE_DOWN_FACTOR
# config.encoder_layers //= SCALE_DOWN_FACTOR
# config.decoder_attention_heads //= SCALE_DOWN_FACTOR
# config.decoder_ffn_dim //= SCALE_DOWN_FACTOR
# config.decoder_layers //= SCALE_DOWN_FACTOR

# print(config)

model = MBartForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))

if model_args.model_name_or_path:
    model = model.from_pretrained(model_args.model_name_or_path)

model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.tgt_lang)

# model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(data_args.tgt_lang)

label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)


sacrebleu = evaluate.load("sacrebleu")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = {}
    
    result["sacrebleu"] = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    result["bleu_1"] = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
    result["bleu_2"] = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=2)
    result["bleu_3"] = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=3)
    result["bleu_4"] = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=4)

    result["rouge"] = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    return result


set_seed(training_args.seed)

# %%
training_args.max_steps = -1
training_args.num_train_epochs = 3


# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
)

train_result = trainer.train()
print(train_result)

trainer.save_model()  
trainer.save_metrics("train", train_result.metrics)

predict_results = trainer.predict(
            test_dataset, metric_key_prefix="predict", max_length=training_args.generation_max_length, num_beams=training_args.generation_num_beams
        )

trainer.save_metrics("predict", predict_results.metrics)

if training_args.predict_with_generate:
    predictions = tokenizer.batch_decode(
        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))

print(predict_results)

