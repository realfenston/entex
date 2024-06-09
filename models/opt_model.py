import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import click
import numpy as np
import datasets
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_MOLCAP_NO_FORMAT,
    PROMPT_MOLCAP_FORMAT,
    PROMPT_MOLGEN_FORMAT,
    PROMPT_CONFGEN_FORMAT,
    PROMPT_POC_MOLGEN_FORMAT,
    PROMPT_MOL_POCGEN_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY,
    RESPONSE_KEY_NL,
    mol_selected_property_names,
    mol_property_names,
    MOLCAP_INPUT_KEY,
    MOLGEN_INPUT_KEY,
    POC_MOLGEN_INPUT_KEY,
    CONFGEN_SMILES_INPUT_KEY,
    CONFGEN_DESC_INPUT_KEY,
    POC_S,
    POC_E,
    MOL_S,
    MOL_E,
    SMI_S,
    SMI_E,
    MOLCAP_INSTRUCTION_LIST,
    MOLGEN_INSTRUCTION_LIST,
    CONFGEN_INSTRUCTION_LIST,
    POC_MOLGEN_INSTRUCTION_LIST,
    MOL_POCGEN_INSTRUCTION_LIST,
)
from mol_vq_dataset import (
    MolVQDataset, 
    MolPocVQDataset,
    sample_molcap_instruction,
    sample_molgen_instruction,
    sample_poc_molgen_instruction,
    sample_confgen_instruction,
    get_inject_vq_fun,
    MOL_REPLACE_TEMPLATE,
    POCKET_REPLACE_TEMPLATE,
    #VQ_CODE_BOOK_SIZE, 
)
import jsonlines
from peft import PeftModel,PeftConfig

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).parent.parent
def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]):
        batch = super().torch_call(examples)
        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)
        response_nl_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)
        labels = batch["labels"].clone()
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[1])[0]:
                response_token_ids_start_idx = idx
                break
            if response_token_ids_start_idx is None:
                for idx in np.where(batch["labels"][i] == response_nl_token_ids[1])[0]:
                    response_token_ids_start_idx = idx
                    break
            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )
            response_token_ids_end_idx = response_token_ids_start_idx + 1
            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100
        # print(labels)
        # import ipdb;ipdb.set_trace()
        batch['labels'] = labels
        return batch


def train(
    *,
    input_model: str,
    local_output_dir: str,
    dbfs_output_dir: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    deepspeed: str,
    gradient_checkpointing: bool,
    local_rank: str,
    bf16: bool,
    logging_steps: int,
    eval_steps: int,
    test_size: Union[float, int],
    save_total_limit: int,
    warmup_steps: int,
    stage: int,
):
    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained("/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/opt-125M/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/opt-125M/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ['<end>', '<instruction>', '<response>', '<response_nl>', '<molecule>', '</molecule>', '<smiles>', '</smiles>', '<pocket>', '</pocket>']})
    model.resize_token_embeddings(len(tokenizer) + 1024)

    # Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
    # The configuraton for the length can be stored under different names depending on the model.  Here we attempt
    # a few possible names we've encountered.
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")
    print(max_length)
    # load train dataset
    molcap_train = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_test2/molcap_tokens_opt.jsonl')
    molgen_train = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_test2/molgen_tokens_opt.jsonl')
    confgen_train = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_test2/confgen_tokens_opt.jsonl')
    poc_molgen_train = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_test2/poc_molgen_tokens_opt.jsonl')
    mol_pocgen_train = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_test2/mol_pocgen_tokens_opt.jsonl')
    dataset_train = molcap_train + molgen_train + confgen_train + poc_molgen_train + mol_pocgen_train
    dataset_train = dataset_train
    
    molcap_valid = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_for_valid/molcap_tokens_opt.jsonl')
    molgen_valid = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_for_valid/molgen_tokens_opt.jsonl')
    confgen_valid = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_for_valid/confgen_tokens_opt.jsonl')
    poc_molgen_valid = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_for_valid/poc_molgen_tokens_opt.jsonl')
    mol_pocgen_valid = read_jsonl('/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/train_llm/opt_data_for_valid/mol_pocgen_tokens_opt.jsonl')
    dataset_valid = molcap_valid + molgen_valid + confgen_valid + poc_molgen_valid + mol_pocgen_valid

    data_collator = DataCollatorForCompletionOnlyLM(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=True,
        # bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=False,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=False,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
    )

    trainer = Trainer(
        model=model.to('cuda'),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    if dbfs_output_dir:
        logger.info(f"Saving Model to {dbfs_output_dir}")
        trainer.save_model(output_dir=dbfs_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--input-model", type=str, help="Input model to fine tune", default=DEFAULT_INPUT_MODEL)
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True, default="/mnt/cc/0_MMGenerateMol/MMGMol_v1/MMMolGen_V1/llm_output_play")
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=300, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=32, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=32, help="Batch size to use for evaluation.")
@click.option(
    "--test-size", type=int, default=100, help="Number of test records for evaluation, or ratio of test records."
)
@click.option("--warmup-steps", type=int, default=1000, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=100, help="How often to run evaluation on test records")
# @click.option("--save-steps", type=int, default=400, help="How often to checkpoint the model")
@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
@click.option("--stage", type=int, help="Training stage to run.")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
