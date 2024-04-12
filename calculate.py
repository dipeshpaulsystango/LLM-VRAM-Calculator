import requests

# Define predefined values
data_types = {'F32': 4, 'FP16': 2, 'Int8': 1, 'Int4': 0.5, 'nf4': 0.5}


def compute_model_size(model_size, dtype):
    # Check if dtype is valid
    if dtype not in data_types:
        print("Invalid dtype. Choose one of F32, FP16, Int8, Int4, nf4.")
        return

    # Remove 'b' from model size input and convert it to lowercase
    model_size = model_size.lower().rstrip('b')

    # Convert model size to integer
    try:
        model_size = int(model_size)
    except ValueError:
        print("Invalid model size. Please enter a valid number.")
        return

    # Compute model size
    base_loaded_model = model_size * 1e9 * data_types[dtype]
    return base_loaded_model


def compute_training_model_size(loaded_model_size):
    # Loaded Model
    loaded_model_gb = loaded_model_size / (1024 ** 3)

    # Gradient, Optimizer (Adam), Activation
    gradient = loaded_model_gb
    optimizer = loaded_model_gb * 2
    activation = loaded_model_gb

    # Training Model Size
    training_model_size = loaded_model_gb + gradient + optimizer + activation
    return training_model_size


def compute_peft_model_size(loaded_model_size, lora_trainable_params_input, dtype):
    # LORA Loaded Model
    lora_loaded_model_gb = (lora_trainable_params_input * 1e9 * data_types[dtype]) / (1024 ** 3)

    # Model In GB
    # Gradient, Optimizer (Adam), Activation
    gradient = lora_loaded_model_gb
    optimizer = lora_loaded_model_gb * 2
    activation = lora_loaded_model_gb

    # Training Model Size
    trainable_peft_model_size = lora_loaded_model_gb + gradient + optimizer + activation
    print(f"Lora Model with {lora_trainable_params_input} parameter in {dtype}: ", trainable_peft_model_size,
          "GB approx")
    return trainable_peft_model_size


# User input
model_size_input = input("Enter Model Size in billion (e.g.: 7b, 13b, 70b) [Default=7b]: ") or '7b'
lora_trainable_params_input = input(
    "Enter LORA Trainable Parameters in billion (e.g.: 0.5b, 0.3b) [Default=0.5b]: ") or '0.5b'
lora_trainable_params_input = float(lora_trainable_params_input.lower().rstrip('b'))
model_name_huggingface = input(
    "Model Name (e.g.: 'teknium/OpenHermes-2.5-Mistral-7B') [Default='teknium/OpenHermes-2.5-Mistral-7B']: ") or 'teknium/OpenHermes-2.5-Mistral-7B'
model_url = f"https://huggingface.co/{model_name_huggingface}/raw/main/config.json"
model_config = requests.get(model_url).json()
seq_length = int(input("Enter Sequence Length (e.g.: 1024, 2028, 4096 etc.) [Default=4096]: ") or 4096)
batch = int(input("Enter Batch Size (1, 2, 4, 6, 8) [Default=1]: ") or 1)

for dtype in ['F32', 'FP16', 'Int8', 'Int4', 'nf4']:
    # Compute base loaded model size
    base_loaded_model_size = compute_model_size(model_size_input, dtype)

    if base_loaded_model_size:
        print("=" * 30 + f" Name: {model_name_huggingface} | dtype: {dtype} | seq_length: {seq_length} | batch: {batch} " + "=" * 30)
        # print(f"Base {model_size_input}b Model Loaded in {dtype} Size: {base_loaded_model_size} bytes")
        # print(
        #     f"Base {model_size_input}b Model Loaded in {dtype} Size: {base_loaded_model_size / (1024 ** 3)} GB approx")

        # Compute training model size
        training_model_size = compute_training_model_size(base_loaded_model_size)
        # print(f"Training Model in GB SFT = {training_model_size} GB approx")

        # Compute PEFT model size
        peft_model_size = compute_peft_model_size(base_loaded_model_size, float(lora_trainable_params_input), dtype)

        # kv = 2 * precision # n_layers * d_model * seq_length * batch
        kv_cache = (2 * data_types[dtype] * model_config.get('num_hidden_layers') * model_config.get(
            'hidden_size') * seq_length * batch) / (1024 ** 3)
        print("kv_cache: ", kv_cache, "GB")
        # print(f"PEFT Model in GB = {peft_model_size} GB approx")
        print(
            f"TOTAL Training Size of Model {model_size_input}b Model Loaded in FP32 (No QUANT) Size with Full Train [WITH GPU OVERHEAD and kv_cache]:",
            training_model_size * 1.2 + kv_cache, "GB")
        print(
            f"TOTAL Training Size of Model {model_size_input}b Model Loaded in {dtype} (QUANT) Size with QLORA [WITH GPU OVERHEAD and kv_cache]:",
            (peft_model_size + base_loaded_model_size / (1024 ** 3)) * 1.2 + kv_cache, "GB")
        print(
            f"TOTAL Training Size of Model {model_size_input}b Model Loaded in FP32 (No QUANT) Size with Full Train [WITHOUT GPU OVERHEAD and kv_cache]:",
            training_model_size + kv_cache, "GB")
        print(
            f"TOTAL Training Size of Model {model_size_input}b Model Loaded in {dtype} (QUANT) Size with QLORA [WITHOUT GPU OVERHEAD and kv_cache]:",
            peft_model_size + base_loaded_model_size / (1024 ** 3) + kv_cache, "GB")
        print(f"TOTAL Inference Size of Model {model_size_input}b Model Loaded in FP32 (No QUANT):",
              compute_model_size(model_size_input, dtype='FP16') / (1024 ** 3), "GB")
        print(f"TOTAL Inference Size of Model {model_size_input}b Model Loaded in {dtype} (QUANT):",
              base_loaded_model_size / (1024 ** 3), "GB")

        print(f"TOTAL Inference Size of Model {model_size_input}b Model Loaded in FP32 (No QUANT):",
              compute_model_size(model_size_input, dtype='FP16') / (1024 ** 3), "GB")
        print(f"TOTAL Inference Size of Model {model_size_input}b Model Loaded in {dtype} (QUANT):",
              base_loaded_model_size / (1024 ** 3), "GB")
print("-" * 50 + ": NOTE ADD 20GB-30GB Extra for Safe Scenario :" + "-" * 50)
