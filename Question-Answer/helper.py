from pynvml import *
from transformers import set_seed

def generate_and_print_answer_using_base_model(dataset, model, tokenizer, index):
    """
    Generate a answer using the provided model and dataset at the specified index, and print the results.

    Args:
        dataset: The dataset containing question and answer pairs.
        model: The pretrained model used for text generation.
        tokenizer: The tokenizer used for tokenization.
        index (int): The index of the question and answer pair to use from the dataset.
    """
    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)

    # Get the question and answer from the dataset
    prompt = dataset['test'][index]['question']
    answer = dataset['test'][index]['answer']

    # Format the prompt for model input
    formatted_prompt = f"Instruct: Answer the below question.\n{prompt}\nOutput:\n"

    # Generate text using the original model based on the formatted prompt
    res = generate_text(model, tokenizer, formatted_prompt, 100)

    # Extract the output from the generated text
    output = res[0].split('Output:\n')[1]

    # Create a dashed line for visual separation
    dash_line = '-'.join('' for x in range(100))

    # Print the results
    print(dash_line)
    print(f'INPUT PROMPT:\n{formatted_prompt}')
    print(dash_line)
    print(f'BASELINE ANSWER:\n{answer}\n')
    print(dash_line)
    print(f'MODEL GENERATION-:\n{output}')

def generate_and_print_answer_using_peft(dataset, model, tokenizer, index):
    """
    Generate a answer using the provided model and dataset at the specified index, and print the results.

    Args:
        dataset: The dataset containing question and answer pairs.
        model: The pretrained model used for text generation.
        tokenizer: The tokenizer used for tokenization.
        index (int): The index of the question and answer pair to use from the dataset.
    """
    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)

    # Get the question and answer from the dataset
    prompt = dataset['test'][index]['question']
    answer = dataset['test'][index]['answer']

    # Format the prompt for model input
    formatted_prompt = f"Instruct: Answer the below question.\n{prompt}\nOutput:\n"

    # Generate text using the original model based on the formatted prompt
    res = generate_text(model, tokenizer, formatted_prompt, 100)

    # Extract the output from the generated text
    output = res[0].split('Output:\n')[1]
    prefix, success, result = output.partition('###')

    # Create a dashed line for visual separation
    dash_line = '-'.join('' for x in range(100))

    # Print the results
    print(dash_line)
    print(f'INPUT PROMPT:\n{formatted_prompt}')
    print(dash_line)
    print(f'BASELINE ANSWER:\n{answer}\n')
    print(dash_line)
    print(f'MODEL GENERATION-:\n{prefix}')


def print_gpu_utilization():
    """
    Print the GPU memory occupied by the GPU device with index 0.

    Uses the NVIDIA Management Library (NVML) to query GPU memory information.
    """
    # Initialize NVML
    nvmlInit()

    # Get handle to the GPU device with index 0
    handle = nvmlDeviceGetHandleByIndex(0)

    # Get memory information for the GPU device
    info = nvmlDeviceGetMemoryInfo(handle)

    # Print GPU memory occupied (converted to MB)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")


def generate_text(model, tokenizer, input_prompt, max_length=100, use_sampling=True):
    """
    Generate text using the provided model based on the input prompt.

    Args:
        model: The model used for text generation.
        input_prompt (str): The input prompt text.
        tokenizer: Tokenizer function used to tokenize the input prompt.
        max_length (int): Maximum length of the generated text (default is 100 tokens).
        use_sampling (bool): Whether to use sampling during generation (default is True).

    Returns:
        list of str: List of generated text sequences.
    """
    # Tokenize the input prompt
    encoded_prompt = tokenizer(input_prompt, return_tensors="pt")

    # Generate text using the model
    generated_tokens = model.generate(
        **encoded_prompt.to("cuda"),  # Move the tokens to GPU for processing
        max_new_tokens=max_length,  # Maximum number of tokens in the generated text
        do_sample=use_sampling,  # Whether to use sampling during generation
        num_return_sequences=1,  # Number of generated sequences to return
        temperature=0.1,  # Sampling temperature (controls randomness)
        num_beams=1,  # Number of beams for beam search (1 means greedy decoding)
        top_p=0.95,  # Top-p sampling threshold (controls diversity)
    ).to('cpu')  # Move the generated sequences back to CPU

    # Decode the generated sequences and remove special tokens
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return generated_texts


def format_and_concatenate_prompt(sample):
    """
    Format various fields of the sample ('instruction', 'output').
    Then concatenate them using two newline characters.
    
    :param sample: Dictionary containing information about the sample
    :return: Modified sample with formatted prompt
    """

    # Constants for different sections of the prompt
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_HEADER = "### Instruct:Summarize the below conversation."
    RESPONSE_HEADER = "### Output:"
    END_HEADER = "### End"

    # Adding introductory blurb
    intro_blurb = f"\n{INTRO_BLURB}"

    # Adding instruction header
    instruction_header = f"{INSTRUCTION_HEADER}"

    # Adding input context if available
    input_context = f"{sample['question']}" if sample["question"] else None

    # Adding response header and the actual answer
    response_header = f"{RESPONSE_HEADER}\n{sample['answer']}"

    # Adding end header
    end_header = f"{END_HEADER}"

    # Creating a list of parts and filtering out None values
    parts = [part for part in [intro_blurb, instruction_header, input_context, response_header, end_header] if part]

    # Joining the parts with two newline characters
    formatted_prompt = "\n\n".join(parts)

    # Updating the 'text' field in the sample with the formatted prompt
    sample["text"] = formatted_prompt

    return sample

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes a batch of text using the provided tokenizer.
    
    Args:
        batch (dict): A dictionary containing text data to be tokenized. Each sample is represented by a string under the key "text".
        tokenizer: An instance of a tokenizer object, typically from the Hugging Face `transformers` library.
        max_length (int): The maximum length of the tokenized sequences. Tokens exceeding this length will be truncated.
    
    Returns:
        dict: Tokenized batch of text data, including token IDs, attention masks, etc.
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def get_max_length(model):
    """
    Determine the maximum length of sequences that can be processed by the given model.

    Args:
        model: Model object.

    Returns:
        int: Maximum length of sequences.
    """
    conf = model.config
    max_length = None

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break

    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")

    return max_length

def print_number_of_trainable_model_parameters(model):
    """
    Calculate and print the number of trainable model parameters, total model parameters,
    and the percentage of trainable parameters out of the total parameters.

    Args:
        model: Model object.

    Returns:
        str: A string containing the information about model parameters.
    """
    trainable_model_params = 0
    all_model_params = 0
    
    for _, param in model.named_parameters():
        all_model_params += param.numel() #numel stands for number of elements
        if param.requires_grad:
            trainable_model_params += param.numel()
    
    trainable_percentage = 100 * trainable_model_params / all_model_params
    
    return f"Trainable model parameters: {trainable_model_params}\nAll model parameters: {all_model_params}\nPercentage of trainable model parameters: {trainable_percentage:.2f}%"


