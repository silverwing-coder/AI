import tiktoken

# Load the correct encoding for your model (e.g., gpt-4, gpt-4o, gpt-3.5-turbo)
encoding = tiktoken.encoding_for_model("gpt-4o")

# Define your text
text = "Hello world, I am learning to count tokens!"

# Turn the text into a list of token integers
tokens = encoding.encode(text)

# Print the tokens
print(tokens)

# Print the number of tokens
print(len(tokens))