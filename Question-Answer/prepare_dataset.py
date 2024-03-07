# Contains Function to Preprocess Dataset  
from functools import partial
from transformers import AutoTokenizer
from datasets import DatasetDict
from helper import format_and_concatenate_prompt, preprocess_batch

def split_dataset(dataset):
    # Define the sizes of train, validation, and test sets (as percentages)
    train_percentage = 0.8
    validation_percentage = 0.1
    test_percentage = 0.1

    # Calculate the number of samples for each set
    num_samples = 20000
    num_train_samples = int(num_samples * train_percentage)
    num_validation_samples = int(num_samples * validation_percentage)
    num_test_samples = num_samples - num_train_samples - num_validation_samples

    # Create empty lists to store indices of train, validation, and test samples
    train_indices = []
    validation_indices = []
    test_indices = []

    # Assign indices to train, validation, and test sets
    indices = list(range(num_samples))
    for i in range(num_train_samples):
        index = indices.pop(0)
        train_indices.append(index)

    for i in range(num_validation_samples):
        index = indices.pop(0)
        validation_indices.append(index)

    for i in range(num_test_samples):
        index = indices.pop(0)
        test_indices.append(index)

    # Create new dataset objects for train, validation, and test sets
    train_dataset = dataset['train'].select(train_indices)
    validation_dataset = dataset['train'].select(validation_indices)
    test_dataset = dataset['train'].select(test_indices)

    # Create a new dataset object using the DatasetDict class
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    return dataset

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
        remove_columns=["question", "answer"],
    )

    # Filter out samples that have input_ids exceeding max_token_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_token_length)

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=shuffle_seed)

    return dataset


