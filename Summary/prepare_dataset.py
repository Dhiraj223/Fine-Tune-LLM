# Contains Function to Preprocess Dataset  
from functools import partial
from transformers import AutoTokenizer
from helper import format_and_concatenate_prompt, preprocess_batch

# SOURCE: https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_token_length: int, shuffle_seed, dataset):
    """
    Preprocesses the dataset by adding prompts, tokenizing, removing unnecessary columns, and shuffling.

    Args:
        tokenizer (AutoTokenizer): Model Tokenizer.
        max_token_length (int): Maximum number of tokens to emit from tokenizer.
        shuffle_seed: Seed value for shuffling the dataset.
        dataset: Input dataset to be preprocessed.

    Returns:
        Hugging Face Dataset: Preprocessed dataset ready for training.
    """
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(format_and_concatenate_prompt)#, batched=True)

    # Define the preprocessing function with specified parameters
    preprocessing_func = partial(preprocess_batch, max_length=max_token_length, tokenizer=tokenizer)

    # Apply the preprocessing function to each batch of the dataset and remove unnecessary columns
    dataset = dataset.map(
        preprocessing_func,
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary'],
    )

    # Filter out samples that have input_ids exceeding max_token_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_token_length)

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=shuffle_seed)

    return dataset


