## NLPDL 2022 Final Project: Domain Specific NLP

2022.12.31

### Dataset

[ACL-ARC and SciERC](https://drive.google.com/drive/folders/1Rc_15j3VwnFChzzKj21qIw9lG1UmlBOn?usp=share_link)

[arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

### Environment

For environment configuration, simply run:

```
pip install -r requirements.txt
```

### Training

An example for MLM post-training:

```
python scripts/train_mlm.py \
	--model_name_or_path path/for/pretrained/model    \
	--output_dir output/path/    \
	--task_adaptation acl/scierc/both    \
	--do_train True    \
	--num_train_epochs 50    \
	--per_device_train_batch_size 64    \
	--max_seq_length 128    \
	--seed 20
```

An example for finetuning:

```
python scripts/train.py \
	--model_name_or_path path/for/pretrained/model    \
	--output_dir output/path/    \
	--do_train True --do_predict True    \
	--do_test_acl True --do_test_sci True    \
	--dataset_name acl/scierc/both    \
	--num_train_epochs 12    \
	--per_device_train_batch_size 64    \
	--per_device_eval_batch_size 64     \
	--seed 20
```

### Checkpoint

The best checkpoint is uploaded on [PKU disk](). Feel free to download!