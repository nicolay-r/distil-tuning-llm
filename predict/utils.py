from bulk_chain.api import iter_content
from bulk_chain.core.utils import dynamic_init

from cfg import SUMMARIZE_PROMPT_LOCALE
from keys import HF_API_KEY
from predict.cfg_multiclinsum import MULTICLINSUM_SUBMISSIONS


def infer_summary(args, input_dicts, lang):
    return iter_content(
        schema={"schema": [{"prompt": SUMMARIZE_PROMPT_LOCALE[lang] + ": {input}", "out": "summary"}]},
        llm=dynamic_init(class_filepath="providers/huggingface_qwen.py", class_name="Qwen2")(
            api_token=HF_API_KEY,
            model_name=MULTICLINSUM_SUBMISSIONS[args.run_id],
            temp=0.1,
            use_bf16=True,
            max_new_tokens=args.max_tokens,
            device=args.device
        ),
        infer_mode="batch",
        batch_size=args.batch_size,
        return_mode="record",
        input_dicts_it=input_dicts,
    )
