<h5>Edited by Sangmork Park at VMI. Last update: Sep. 2026</h5>

<h2>Programming Environment ....</h2>
<h3>Authentication (TOKEN submission) </h3>
<ol>
  <li>Save to token in "secrets"</li>
  <li>Use access token </li>

  ```py
    import os
    from google.colab import userdata
    userdata.get('HF_TOKEN')
  ```
</ol>

<h3>Performance improvement </h3>
<ol>
  <li>Switch to GPU mode to save execution time: "Runtime >> Change runtime type"</li>
  <li>(Optional) Move your model and inputs to the GPU </li>

  ```py
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=100)
  ```
  <li>(Optional) Load the model in lower precision: Quantization </li>

  ```py
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
      model="your-model-id",
      quantization_config=quantization_config
    )
  ```
</ol>
