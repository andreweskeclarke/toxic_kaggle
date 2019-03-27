# Kaggle: Toxic Kaggle Comments with BERT
Playing with BERT using the Kaggle Toxic Comment contest.

Inspired by the BERT repo https://github.com/google-research/bert and by https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d


# Running

I run everything inside a docker container:

```
sudo docker run --rm -it --runtime=nvidia -p 0.0.0.0:6006:6006 -v $PWD:$PWD -it tensorflow/tensorflow:latest-gpu-py3
pip install -r requirements.txt
```

First conver the records:

```
./convert_to_tf_records.py --input_file data/train.csv --output_training_file data/train.tf_record --output_validation_file data/validation.tf_record --do_lower_case --validation_ratio 0.1
```

Then run the transfer training:

```
./run_training.py
```
