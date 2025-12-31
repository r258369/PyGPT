def encode_newline(inp):
    return inp.replace('\n', '<N>')

def decode_newlines(inp):
    return inp.replace('<N>', '\n')


from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("GPyT/checkpoint-20000")
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer")

tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})
model.resize_token_embeddings(len(tokenizer))
inp = """import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

#pie plot
"""
inp = encode_newline(inp)


#num_beams=10,
#temperature=0.7,
#no_repeat_ngram_size=5,
#num_return_sequences=3,
input_ids = tokenizer.encode(inp, return_tensors='pt')
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
    pad_token_id=tokenizer.pad_token_id
)

sequence = beam_output['sequences'][0] 
out = decode_newlines(tokenizer.decode(sequence, skip_special_tokens=True))


print("--------------------------------------------")
print("Input:")
auto_complete = ""
split = out.split('\n')
newlinecount = inp.count("<N>")

for s in split[:newlinecount]:
  auto_complete += s+'\n'
print(auto_complete)
print("--------------------------------------------")
print("Output:")
print(out)
print()

