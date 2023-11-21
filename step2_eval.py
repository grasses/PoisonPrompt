import logging
import os
import os.path as osp
import sys
import numpy as np
from typing import Dict

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args

from tasks.utils import *

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)



def evaluate(args, trainer, checkpoint=None):
    logger.info("*** Evaluate ***")
    print(f"=============> checkpoint:{checkpoint}")

    trainer.resume_from_checkpoint = checkpoint
    trainer._load_from_checkpoint(resume_from_checkpoint=checkpoint)
    trainer.args.trigger = args.trigger
    trainer.trigger_ids = torch.tensor(args.trigger, device=trainer.device).long()

    score, asr = trainer.evaluate_backdoor(synonyms_trigger_swap=True)
    metrics = trainer.evaluate(ignore_keys=["hidden_states", "attentions"])
    metrics["asr"] = asr
    metrics["score"] = score
    trainer.evaluate_clean()

    path = f"{args.output_dir}/exp_attentions.pth"
    torch.save(trainer.eval_memory, path)
    print(f"-> save exp_attentions to:{path}")

    trainer.log_metrics("eval", metrics)
    path = osp.join(args.output_dir, "exp_acc_asr.pth")
    torch.save(metrics, path)
    print(f"-> save exp_acc_asr to:{path}")


if __name__ == '__main__':
    args = get_args()
    assert args[2].trigger is not None
    assert args[2].use_checkpoint is not None

    trigger = args[2].trigger
    #path = osp.join(args[2].use_checkpoint, "args.pt")
    #args = torch.load(path)
    #args[2].trigger = trigger
    #print(f"-> load args from: {path} trigger:{args[2].trigger}")

    p_type = "prefix" if args[0].prefix else "prompt"
    output_root = osp.join("checkpoints", f"{args[1].task_name}_{args[1].dataset_name}_{args[0].model_name_or_path}_{args[2].backdoor}_{p_type}")
    output_dir = osp.join(output_root, f"t{args[2].trigger_num}_p{args[2].poison_rate:0.2f}")
    model_args, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.dataset_name.lower() in NER_DATASETS
        from tasks.ner.get_trainer import get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.dataset_name.lower() in SRL_DATASETS
        from tasks.srl.get_trainer import get_trainer

    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from tasks.qa.get_trainer import get_trainer
    elif data_args.task_name.lower() == "ag_news":
        from tasks.ag_news.get_trainer import get_trainer
    elif data_args.task_name.lower() == "imdb":
        from tasks.imdb.get_trainer import get_trainer
    else:
        raise NotImplementedError(
            'Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)
    trainer, predict_dataset = get_trainer(args)

    last_checkpoint = osp.join(training_args.use_checkpoint, "checkpoint")
    print(f"-> last_checkpoint:{last_checkpoint} trigger:{training_args.trigger}")
    evaluate(training_args, trainer, checkpoint=last_checkpoint)


















