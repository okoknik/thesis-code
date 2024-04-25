# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#from collections import defaultdict
import pickle
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
#import sys
from typing import (
    Optional,
    Dict,
  #  Sequence,
    Any,
 #   Union,
    List
)
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
#import importlib
#from packaging import version
#from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding

)
from datasets import (
    Dataset,
    Features,
    ClassLabel,
    Value
)

from sklearn.metrics import (
    top_k_accuracy_score,
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from peft import (
    prepare_model_for_kbit_training,
    AdaLoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from peft.tuners.ia3 import IA3Layer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[MASK]"

# insert weightst for weighted CE here as list of weights where position in list coincides with label
class_weights = torch.Tensor(np.load("class-weights.npy")).to(device="cuda:0")

id2label={0: 'Unemployment Rate',
 1: 'Monetary Policy',
 2: 'National Budget',
 3: 'Tax Code',
 4: 'Industrial Policy',
 5: 'Minority Discrimination',
 6: 'Gender Discrimination',
 7: 'Age Discrimination',
 8: 'Handicap Discrimination',
 9: 'Voting Rights',
 10: 'Freedom of Speech',
 11: 'Right to Privacy',
 12: 'Anti-Government',
 13: 'Health Care Reform',
 14: 'Insurance',
 15: 'Medical Facilities',
 16: 'Insurance Providers',
 17: 'Manpower',
 18: 'Disease Prevention',
 19: 'Infants and Children',
 20: 'Long-term Care',
 21: 'Drug and Alcohol Abuse',
 22: 'R&D',
 23: 'Trade',
 24: 'Subsidies to Farmers',
 25: 'Food Inspection & Safety',
 26: 'Animal and Crop Disease',
 27: 'R&D',
 28: 'Worker Safety',
 29: 'Employment Training',
 30: 'Employee Benefits',
 31: 'Labor Unions',
 32: 'Fair Labor Standards',
 33: 'Youth Employment',
 34: 'Higher',
 35: 'Elementary & Secondary',
 36: 'Underprivileged',
 37: 'Vocational',
 38: 'Excellence',
 39: 'Drinking Water',
 40: 'Waste Disposal',
 41: 'Hazardous Waste',
 42: 'Air Pollution',
 43: 'Species & Forest',
 44: 'Land and Water Conservation',
 45: 'Nuclear',
 46: 'Electricity',
 47: 'Coal',
 48: 'Alternative & Renewable',
 49: 'Conservation',
 50: 'Immigration',
 51: 'Mass',
 52: 'Highways',
 53: 'Air Travel',
 54: 'Railroad Travel',
 55: 'Maritime',
 56: 'Infrastructure',
 57: 'Agencies',
 58: 'White Collar Crime',
 59: 'Illegal Drugs',
 60: 'Court Administration',
 61: 'Juvenile Crime',
 62: 'Child Abuse',
 63: 'Family Issues',
 64: 'Criminal & Civil Code',
 65: 'Crime Control',
 66: 'Police',
 67: 'Low-Income Assistance',
 68: 'Elderly Assistance',
 69: 'Disabled Assistance',
 70: 'Volunteer Associations',
 71: 'Child Care',
 72: 'Community Development',
 73: 'Urban Development',
 74: 'Rural Development',
 75: 'Low-Income Assistance',
 76: 'Elderly',
 77: 'Banking',
 78: 'Securities & Commodities',
 79: 'Corporate Management',
 80: 'Small Businesses',
 81: 'Copyrights and Patents',
 82: 'Consumer Safety',
 83: 'Alliances',
 84: 'Nuclear Arms',
 85: 'Military Aid',
 86: 'Personnel Issues',
 87: 'Foreign Operations',
 88: 'Telecommunications',
 89: 'Broadcast',
 90: 'Computers',
 91: 'R&D',
 92: 'Trade Agreements',
 93: 'Competitiveness',
 94: 'Foreign Aid',
 95: 'Resources Exploitation',
 96: 'Developing Countries',
 97: 'International Finance',
 98: 'Western Europe',
 99: 'Specific Country',
 100: 'Human Rights',
 101: 'Organizations',
 102: 'Diplomats',
 103: 'Intergovernmental Relations',
 104: 'Bureaucracy',
 105: 'Employees',
 106: 'Property Management',
 107: 'Branch Relations',
 108: 'Political Campaigns',
 109: 'National Parks'}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default= "xlm-roberta-large" 
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=10000,#106567, #=112175 - (0.5*112175)
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None, #0.5*112175
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=256,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    train_dataset: str = field(
        default='train.csv',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    test_dataset: str = field(
        default='test.csv',
        metadata={"help": "Which dataset to test on. See datamodule for options."}
    )
    predict_dataset: str = field(
        default='predict_data/9cut_txt.csv',
        metadata={"help": "Which data to predict on."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [csv]"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    max_memory_MB: int = field(
        default=12000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='wandb',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    num_labels: int = field(
        default=110,
        metadata={"help":'Number of classes in dataset.'}
    )
    output_dir: str = field(default='./output_cls', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='adamw_8bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=8, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    #per_device_eval_batch_size: int = field(default=24, metadata={"help":'The evaluation/prediction batch sizer per GPU. Change if out of memory in evaluation/prediction.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10543, metadata={"help": 'How many optimizer update steps to take.'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=20, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

#define custom Trainer for weighted CE-Loss
class CustomTrainer(Trainer):
    #adapted huggingface impl
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        if isinstance(outputs, dict) and "logits" not in outputs:
            raise ValueError(
                "The model did not return logits from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        logits = outputs["logits"] #if isinstance(outputs, dict) else outputs[0]
        
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = loss_fct(logits.view(-1,110), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    print(f"Compute dtype: {compute_dtype}")
    
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    if args.do_eval or args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels,
            id2label=id2label,
            label2id=dict((v,k) for k,v in id2label.items()),
            problem_type="single_label_classification",
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        )
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'))

            setattr(model, 'model_parallel', True)
            setattr(model, 'is_parallelizable', True)

            model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        else:
            print("No checkpoints found.")
            return None
    else:    
                
        model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        num_labels=args.num_labels,
        id2label=id2label,
        label2id=dict((v,k) for k,v in id2label.items()),
        problem_type="single_label_classification",
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=['classifier'],
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        )
        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)

        model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'))

            setattr(model, 'model_parallel', True)
            setattr(model, 'is_parallelizable', True)

            model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        else:
            print(f'adding AdaLora modules...')
            config = AdaLoraConfig(
                    target_modules=["value","key", "query"],
                    r=8,
                    lora_alpha=32,
                    task_type=TaskType.SEQ_CLS,
                    modules_to_save=['classifier']
                )
            model = get_peft_model(model, config)
            
    # change dtypes of layers
    for name, module in model.named_modules():
        if isinstance(module, IA3Layer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'classifier' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    print(model)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    """
    # train data
    if args.do_train:
        try:
            features = Features({"text": Value("string"), "labels": ClassLabel(num_classes=args.num_labels)})
            if args.train_dataset.endswith('.csv'):
                train_dataset = Dataset.from_pandas(pd.read_csv(args.train_dataset),features=features)

                if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
                    train_dataset = train_dataset.select(range(args.max_train_samples))

                tk_train_dataset=train_dataset.map(lambda examples: tokenizer(examples["text"]), batched=True,remove_columns=["text"])
            else:
                raise ValueError(f"Unsupported dataset format: {args.train_dataset}")   
        except:
            raise ValueError(f"Error loading dataset from {args.train_dataset}")

        
    # eval data
    if args.do_eval: 
        if args.test_dataset.endswith('.csv'):
            try:
                features = Features({"text": Value("string"), "labels": ClassLabel(num_classes=args.num_labels)})
                eval_dataset = Dataset.from_pandas(pd.read_csv(args.test_dataset),features=features)
            except:
                raise ValueError(f"Error loading dataset from {args.test_dataset}")                
            if args.do_predict:
                eval_dataset = eval_dataset.remove_columns("labels")

            tk_eval_dataset = eval_dataset.map(lambda examples: tokenizer(examples["text"]), batched=True, remove_columns=["text"])
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            tk_eval_dataset = tk_eval_dataset.select(range(args.max_eval_samples))
        
    if args.do_predict:
        if args.predict_dataset.endswith('.csv'):
            try:
                features = Features({"text": Value("string")})
                predict_dataset = Dataset.from_pandas(pd.read_csv(args.predict_dataset),features=features)
            except:
                raise ValueError(f"Error loading dataset from {args.predict_dataset}")
            tk_predict_dataset = predict_dataset.map(lambda examples: tokenizer(examples["text"],truncation=True), batched=True, remove_columns=["text"])                

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
        #pad_to_multiple_of=8
    )
    return dict(
        train_dataset=tk_train_dataset if args.do_train else None,
        eval_dataset=tk_eval_dataset if args.do_eval else None,
        predict_dataset=tk_predict_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        #if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    #metrics
    def compute_metrics(eval_pred):

        logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        predictions = np.argmax(logits, axis=-1)
       
        
        class_rep = classification_report(y_true=labels,y_pred=predictions,target_names=id2label.values(),output_dict=True)
        b_accuracy = balanced_accuracy_score(y_true=labels,y_pred=predictions)
        top_2_accuracy = top_k_accuracy_score(y_true=labels,y_score=logits,k=2)
        accuracy = accuracy_score(y_true=labels,y_pred=predictions)
        precision = precision_score(y_true=labels,y_pred=predictions,average="weighted")
        recall = recall_score(y_true=labels,y_pred=predictions,average="weighted")
        f1 = f1_score(y_true=labels,y_pred=predictions, average="weighted")
        #roc_auc = roc_auc_score(y_true=labels,y_score=probabilities,average="weighted")
        # save class report
        with open('class_rep.pickle', 'wb') as handle:
            pickle.dump(class_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(class_rep)
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
        return {"precision": precision, "recall": recall, "f1-weighted": f1, 'balanced-accuracy': b_accuracy,"accuracy": accuracy, "top_2_accuracy":top_2_accuracy}#, "roc_auc":roc_auc}
    
    # define trainer
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        compute_metrics=compute_metrics
    )

    # Callbacks
    trainer.add_callback(SavePeftModelCallback)
   
    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    if args.report_to == 'wandb':
      import wandb
      os.environ["WANDB_PROJECT"] = "MasterThesisIA3CLS"  # name your W&B project
      os.environ["WANDB_LOG_MODEL"] = "checkpoint"
      wandb.login()

   

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logging.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        # Edit: https://github.com/huggingface/peft/issues/859
       
        if completed_training:
            train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
            logging.info("*** Resuming from checkpoint. ***")
        else:    
            train_result=trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics) 
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logging.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        all_metrics.update(metrics)
        if training_args.report_to == "wandb":
            wandb.log(all_metrics)
    # Prediction
    if args.do_predict:
        logging.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        #print(predictions)
        predictions = torch.Tensor(predictions)
        probabilities = torch.softmax(predictions, dim=1).tolist()#[0]
        #print("softmax",probabilities)
        #probabilities = {model.config.id2label[index]: round(probability * 100, 2) for index, probability in enumerate(probabilities)}
        #print(probabilities)
        codes = []
        probs = []
        print("Creating code and probabilities df.")
        for v in probabilities:
            result = {model.config.id2label[index]: round(probability * 100, 2) for index, probability in enumerate(v)}
            result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
            codes.append(list(result.items())[0][0])
            probs.append(list(result.items())[0][1])
        #print("rounded",probabilities)
        #probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        #print(codes, probs)
        # here it is needed to record also confidence values and then cut-off low confidences in analysis later
        # record first item in dict (label+probability value)
        # with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
        #     for i, example in enumerate(data_module['predict_dataset']):
        #         example['prediction_with_input'] = predictions[i].strip()
        #         example['prediction'] = predictions[i].replace(example['input'], '').strip()
        #         fout.write(json.dumps(example) + '\n')
        df = pd.DataFrame({"codes":codes,"probs":probs})
        print(df.head())
        df.to_parquet(args.predict_dataset+"predictions.parquet",compression="zstd")
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)

        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
