import sys
sys.path.append("..")

from bulk_chain.api import iter_content
from bulk_chain.core.utils import dynamic_init

from cfg import SUMMARIZE_PROMPT
from predict.keys import HF_API_KEY


content_it = iter_content(
    schema={
        "schema": [
            {"prompt": SUMMARIZE_PROMPT + ": {input}", "out": "summary"}
        ]
    },
    llm=dynamic_init(class_filepath="../predict/providers/huggingface_qwen.py", class_name="Qwen2")(
        api_token=HF_API_KEY,
        model_name="nicolay-r/qwen25-05b-multiclinsum-standard",
        temp=0.1,
        max_new_tokens=1024
    ),
    infer_mode="single",
    return_mode="record",
    input_dicts_it=[
        {"input": "This is a test medical report of patient 62 y.o who is female."}
    ],
)

for data in content_it:
    print(data)
