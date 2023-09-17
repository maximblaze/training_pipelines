import os
import argparse
import pandas as pd
import numpy as np
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from transformers import Trainer
from datasets import load_from_disk, Dataset, ClassLabel, Sequence
import torch
import evaluate
import numpy as np
import logging 
from huggingface_hub import HfFolder


logger = logging.getLogger(__name__)

# Metric Id
metric = evaluate.load("f1")

# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


def training_function(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    if args.train_dataset and args.eval_dataset:
        train_dataset = pd.read_csv(args.train_dataset)
        eval_dataset = pd.read_csv(args.eval_dataset)
        train_dataset = Dataset.from_pandas(train_dataset)
        eval_dataset = Dataset.from_pandas(eval_dataset)
    else:
        train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
        eval_dataset = load_from_disk(os.path.join(args.dataset_path, "eval"))
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Prepare model labels - useful for inference
    label_names = sorted(set(train_dataset["label"]))

    # Cast to ClassLabel
    train_dataset = train_dataset.cast_column("label", Sequence(ClassLabel(names=label_names)))

    labels = train_dataset.features["label"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    # Define training args
    output_dir = args.model_id.split("/")[-1] if "/" in args.model_id else args.model_id
    output_dir = f"{output_dir}-finetuned"

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    eval_res = trainer.evaluate(eval_dataset=eval_dataset)

    print(eval_res)

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(output_dir)
    trainer.create_model_card()
    # Push the results to the hub
    if args.repository_id:
        trainer.push_to_hub()



def create_parser():
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group('Device Targets')
    g.add_argument(
        '--config-file',
        dest='config_file',
        type=argparse.FileType(mode='r'))
    g.add_argument('--name', default=[], action='append')
    g.add_argument('--age', default=[], action='append')
    g.add_argument('--delay', type=int)
    g.add_argument('--stupid', dest='stupid', default=False, action='store_true')
    return parser

def create_parser():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--config-file',
                        dest='text_clf_config.yaml',
                        default='text_clf_config.yaml',
                        type=argparse.FileType(mode='r'))
    
    parser.add_argument("--model_id", type=str, default="bert-large-uncased", help="Model id to use for training.")

    parser.add_argument("--train_dataset", type=str, default=None, help="Dataset with label and text columns")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Dataset with label and text columns")

    parser.add_argument("--train_dataset_path", type=str, default="'/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv'", help="Path to the already processed dataset.")
    parser.add_argument("--eval_dataset_path", type=str, default="'/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv'", help="Path to the already processed dataset.")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    parser.add_argument(
        "--repository_id", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


def main():
    args, _ = parse_args(create_parser())
    training_function(args)


if __name__ == "__main__":
    main()


#python3 transformer_classifier_fine_tuning.py --train_dataset corona_data/train_data.csv --eval_dataset corona_data/eval_data.csv 