import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoAdapterModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from dataHelper import get_dataset
import wandb


# initialize logger
logger = logging.getLogger(__name__)


# define arguments
@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the training data."},
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."},
    )
    do_test_acl: bool = field(
        default=True,
        metadata={},
    )
    do_test_sci: bool = field(
        default=True,
        metadata={},
    )
    do_fs: bool = field(
        default=False,
        metadata={},
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use Adapter"},
    )

    

def main():
    # Step 1: Initialize parser and parse args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    
    # Initialize wandb
    wandb_name = 'baseline_' + data_args.dataset_name
    if model_args.use_adapter:
        wandb_name += '_with_adapter'
    if data_args.do_fs:
        wandb_name += '_fs'
    wandb_name = wandb_name + '_seed' + str(training_args.seed)
    wandb.init(project="NLP_project", name=wandb_name)
    
    
    # Step 2: Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f'Training/evaluation parameters {training_args}')
    
    
    # Step 3: set_seed
    set_seed(training_args.seed)
    
    
    # Step 4: get_dataset
    dataset_name = data_args.dataset_name
    if dataset_name[0] == '[':
        dataset_name = list(dataset_name[1:-1].split(','))
    raw_datasets = get_dataset(dataset_name, training_args.seed, data_args.do_fs)
    label_list = raw_datasets['train'].unique('label')
    num_labels = len(label_list)
    logger.info(f'Built raw datasets {dataset_name}, {num_labels} labels in total')
    
    
    # Step 5: Setup config, tokenizer, and model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool('.ckpt' in model_args.model_name_or_path),
        config=config,
    )
    
    
    # Task 3: Use Adapter
    #model.set_active_adapters("20")
    if model_args.use_adapter:
        adapter_name = wandb_name + '_' + str(training_args.seed)
        model.add_adapter(adapter_name)
        model.train_adapter(adapter_name)
        #model.add_classification_head(adapter_name, num_labels=num_labels)
        model.set_active_adapters(adapter_name)
    
    
    # Step 6: Preprocess the raw dataset
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['text'], padding='max_length', max_length=max_seq_length, truncation=True)
        result['label'] = examples['label']
        return result
    
    raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
    
    
    # Step 7: Evaluation metrics: micro_f1, macro_f1, accuracy
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        
        f1 = evaluate.load('f1')
        micro_f1 = f1.compute(predictions=preds, references=p.label_ids, average='micro')
        macro_f1 = f1.compute(predictions=preds, references=p.label_ids, average='macro')
        acc = evaluate.load('accuracy')
        accuracy = acc.compute(predictions=preds, references=p.label_ids)
        
        result = {
            'micro_f1': micro_f1['f1'],
            'macro_f1': macro_f1['f1'],
            'accuracy': accuracy['accuracy'],
        }
        return result
    
    
    # Step 8: Use DataCollatorWithPadding
    data_collator = default_data_collator
    
    
    # Step 9: Initialize trainer
    train_dataset=raw_datasets["train"]
    eval_dataset=raw_datasets["val"]
    test_dataset=raw_datasets["test"]
    if data_args.do_test_acl:
        test_acl_dataset=raw_datasets["test_acl"]
    if data_args.do_test_sci:
        test_sci_dataset=raw_datasets["test_sci"]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        if model_args.use_adapter:
            model.save_all_adapters(training_args.output_dir)
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        if data_args.do_test_acl:
            predict_dataset = test_acl_dataset
            predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
            metrics["predict_samples"] = len(predict_dataset)
            trainer.log_metrics("predict_acl", metrics)
            trainer.save_metrics("predict_acl", metrics)
        if data_args.do_test_sci:
            predict_dataset = test_sci_dataset
            predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
            metrics["predict_samples"] = len(predict_dataset)
            trainer.log_metrics("predict_sci", metrics)
            trainer.save_metrics("predict_sci", metrics)

if __name__ == '__main__':
    main()

