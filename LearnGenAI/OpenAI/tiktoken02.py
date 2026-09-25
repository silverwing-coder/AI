import tiktoken as tk

def count_tokens_by_encoding(prompt: str, encoding_name: str) -> int:
    encoding = tk.get_encoding(encoding_name)
    encoded_string = encoding.encode(prompt)
    num_tokens = len(encoded_string)
    return num_tokens

def count_tokens_for_model(prompt: str, model_name: str) -> int:
    encoding = tk.encoding_for_model(model_name)
    encoded_string = encoding.encode(prompt)
    num_tokens = len(encoded_string)
    return num_tokens

if __name__ == "__main__":
    prompt = "Hello world, I am learning to count tokens!"

    encoding_name = "cl100k_base"
    num_tokens_encoding = count_tokens_by_encoding(prompt, "cl100k_base")
    print(f"Number of tokens in the prompt for {encoding_name} encoding: {num_tokens_encoding}")

    model_name = "gpt-4o"
    num_tokens_model = count_tokens_for_model(prompt, "gpt-4o")
    print(f"Number of tokens in the prompt for {model_name} model: {num_tokens_model}")