import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed,AutoModelForCausalLM,AutoTokenizer, T5ForConditionalGeneration
import transformers

# tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
# tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
#

# # model = AutoModelForCausalLM.from_pretrained('google/flan-t5-large')
# model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')


# tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
# model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")


model_id = "openchat/openchat-3.6-8b-20240522"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

sentence = "Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies,Summarize the following patient-doctor dialogue: Doctor: Good afternoon, sir. Did you just have a birthday? I don't have my chart with me right now, the nurse is bringing it. Patient: Good afternoon, sir. Yes, I just turned fifty five. Doctor: You identify as African American, correct? Patient: Yes, that's right. Doctor: When was your last visit, sir? Patient: Um, it was on July twenty ninth two thousand eight. Doctor: Yes, I see. Did we go over your M R I results? Patient: No, I was having those new seizures, remember? Doctor: Yes, I do. Well, the M R I demonstrated right contrast temporal mass. Patient: What exactly does that mean, doctor? Doctor: Well, given this mass, and your new seizures, I am concerned that this could be a high grade glioma, we'll need to do more tests. SUMMARY:"
# sentence = "Doctor: Good afternoon, sir. Did you just have a birthday? I don't have my chart with me right now, the nurse is bringing it. Patient: Good afternoon, sir. Yes, I just turned fifty five. Doctor: You identify as African American, correct? Patient: Yes, that's right. Doctor: When was your last visit, sir? Patient: Um, it was on July twenty ninth two thousand eight. Doctor: Yes, I see. Did we go over your M R I results? Patient: No, I was having those new seizures, remember? Doctor: Yes, I do. Well, the M R I demonstrated right contrast temporal mass. Patient: What exactly does that mean, doctor? Doctor: Well, given this mass, and your new seizures, I am concerned that this could be a high grade glioma, we'll need to do more tests."

inputs = tokenizer(sentence, return_tensors="pt").to("cuda")

messages = [
    {"role": "user", "content": "Explain how large language models work in detail."},
]

generator = transformers.pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)


set_seed(42)
# generator(sentence, max_length=1024, num_return_sequences=1, do_sample=True)
# with torch.no_grad():
#     beam_output = model.generate(**inputs,
#                                 min_length=100,
#                                 max_length=1024,
#                                 num_beams=5,
#                                 early_stopping=True
#                                 )
# print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
print(generator(sentence, max_length=1024, num_return_sequences=1, do_sample=True))
