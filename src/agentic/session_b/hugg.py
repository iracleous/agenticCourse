from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Input message
user_input = "Hello! Can you tell me a fun fact about space?"

# Tokenize input
inputs = tokenizer([user_input], return_tensors="pt")

# Generate a response
reply_ids = model.generate(**inputs)
reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

print("BlenderBot:", reply)
