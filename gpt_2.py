
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel,GPT2Tokenizer,DataCollatorForLanguageModeling
#!pip install datasets
from datasets import load_dataset
from transformers import Trainer,TrainingArguments

paths = ["python.txt"]
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths,vocab_size=52_000,min_frequency=2,special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer.save_model("tokenizer")

inp = "print('hello world'!)"
t = tokenizer.encode(inp)
print(t.ids)
print(t.tokens)

tokenizer = GPT2Tokenizer.from_pretrained('tokenizer')
tokenizer.add_special_tokens({
    "eos_token":"</s>",
    "bos_token":"<s>",
    "unk_token":"<unk>",
    "pad_token":"<pad>",
    "mask_token":"<mask>"
})
t = tokenizer.encode(inp)
print(t)
print(tokenizer.decode(t))

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token = tokenizer.bos_token_id,
    eos_token = tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
dataset = load_dataset("text",data_files=paths)


def encode(lines):
  return tokenizer(lines['text'], add_special_tokens=True,truncation=True,max_length=512)
dataset.set_transform(encode)

dataset = dataset['train']
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer,mlm=False,mlm_probability=0.15)


training_args = TrainingArguments(
  output_dir="GPyT",
  overwrite_output_dir=True,
  num_train_epochs=1,
  per_device_train_batch_size=10,
  save_steps= 10_000 ,
  save_total_limit=2,
  prediction_loss_only=True ,
  remove_unused_columns=False
)
trainer = Trainer(
  model=model ,
  args=training_args ,
  data_collator=data_collator,
  train_dataset=dataset,
)

trainer.train()
trainer.save_model("GPyT")
#998471cbdd9cf927efcaf855d9d9eef0bce9e69c
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("GPyT").to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer")

tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})
model.resize_token_embeddings(len(tokenizer))

input_ids = tokenizer.encode(inp, return_tensors='pt').to("cuda")

beam_output = model.generate(
    input_ids,
    max_length=100,
    num_beams=10,
    temperature=0.7,
    no_repeat_ngram_size=5,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id
)

for beam in beam_output:
    out = tokenizer.decode(beam, skip_special_tokens=True)
    fout = out.replace('<N>', '\n')
    print(fout)

def encode_newline(inp):
    return inp.replace('\n', '<N>')

def decode_newlines(inp):
    return inp.replace('<N>', '\n')


from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("GPyT").to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer")

tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})
model.resize_token_embeddings(len(tokenizer))

# Input text
inp = """if x=="""
inp = encode_newline(inp)

input_ids = tokenizer.encode(inp, return_tensors='pt').to("cuda")

# Generate text
beam_output = model.generate(
    input_ids,
    max_length=150,
    num_beams=10,
    temperature=0.7,
    no_repeat_ngram_size=5,
    num_return_sequences=3,
    return_dict_in_generate = True,
    output_scores = True,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id  # Avoid warnings
)

# Access the generated sequence and decode
sequence = beam_output['sequences'][0]  # beam_output is a tensor; take the first sequence
out = decode_newlines(tokenizer.decode(sequence, skip_special_tokens=True))

print(out)
print()
print("--------------------------------------------")
auto_complete = ""
split = out.split('\n')
newlinecount = inp.count("<N>")

for s in split[:newlinecount+1]:
  auto_complete += s+'\n'
print(auto_complete)

"""