## Known issues

* https://github.com/huggingface/transformers/blob/v4.33.2/examples/pytorch/summarization/run_summarization.py#L657
    * ROUGE calculation
* https://github.com/huggingface/evaluate/issues/609
    * `!pip install datasets==3.6.0 evaluate==0.4.3`
* https://github.com/huggingface/transformers/issues/36331
    * `!pip install transformers==4.45.2` from [this workaround](https://discuss.huggingface.co/t/typeerror-sentencetransformertrainer-compute-loss-got-an-unexpected-keyword-argument-num-items-in-batch/114298/4)
* https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941
