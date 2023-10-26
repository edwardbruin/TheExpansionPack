import os
import pandas as pd
from huggingface_hub import hf_hub_download
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import string
import json
from torch.utils.data import Dataset
import random
import re
from datetime import datetime
import html
import numpy as np

def benchmark(model, tokenizer, n=100, checkvalid=False, verbose=False):
    from datetime import datetime as dt
    times = []
    for i in range(n):
        if i% (n//10) == 0:
            print(f'{i*10//n}% complete')
        now = dt.now()
        result = demo(model, tokenizer)
        delta = dt.now()-now
        if checkvalid and result['short'] not in result['predicted']:
            delta = now-dt.now()
        times.append(delta.total_seconds())
        if verbose:
            print(delta)
    return times

def quantized_load(model_id="vilsonrodrigues/falcon-7b-instruct-sharded"):
    import torch
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
    import json
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config
        , trust_remote_code=True
    )
    print("model loaded")
    return(model, tokenizer)

def make_pipeline(model, tokenizer):
    from transformers import pipeline
    pipeline_new = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=150,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, 
        return_full_text=False
    )
    return(pipeline_new)

def process_sequence(text):
    processed = ''
    import re

    pattern = r'\[/?(LONG|SHRT|MESS)\][^\[]*'
    result = re.sub(pattern, '', text, flags=re.DOTALL)
    return result

def demo(model, tokenizer, choose=False):
    import random
    input_file = "./sequentialised.json"
    with open(input_file, 'r') as f:
        data = json.load(f)

    index = random.randrange(len(data))
    selected_data = data[index]
    
    if choose == True:
        print(f"data length: {len(data)}")
        user_input = input(">> select index:")
    
        if user_input.isdigit():
            index = int(user_input)
            if index > 0 and index < len(data):
                selected_data = data[index - 1]
    
    seed, actual = selected_data['text'].split('[LONG]')
    pipeline = make_pipeline(model, tokenizer)
    sequences = pipeline(seed)
    text = sequences[0]['generated_text']
    predicted = process_sequence(text)
    #print(f'seed: {seed}')
    #print(f'predicted: {predicted}')
    #print(f'actual: {actual}')

    pattern = r"\[SHRT\](.*?)\[/SHRT\]"
    short = re.findall(pattern, selected_data['text'])[0]
    
    result = {
        'original': selected_data['text'],
        'seed': seed,
        'short':short,
        'actual':actual,
        'predicted': predicted,
        'index':index
    }
    return(result)

def link_messages(file_path):
    # Define a function to find message pairs with alternating authors.
    def find_author_pairs(data):
        unique_users = set()
        
        for d in data:
            unique_users.add(d["user"])
        #print("Unique users:", unique_users)
        unique_users_list = list(unique_users)
    
        # Print the list with index numbers
        print("Unique users:")
        for index, user in enumerate(unique_users_list):
            print(f"{index + 1}: {user}")
        
        user_input = input(">> select index:")
        if not user_input.isdigit():
            print('not digit')
            return(False)
    
        index = int(user_input)
        if index == 0:
            print('no index')
            return(False)
        
        if index > len(unique_users_list):
            print('no index')
            return(False)
        
        try:
            selected_user = unique_users_list[index - 1]
        except:
            print('no index')
            return(False)
        print(selected_user)
        
        author_a = selected_user
        
        # Initialize an empty list to store the message pairs.
        message_pairs = []
        # Initialize variables to keep track of the current prompt's author and previous author.
        current_prompt = None
        prev_author = None
    
        # Get the author of the first message in the data.
        #author_a = data[0]['user']
        # Get the content of the first message in the data.
        prev_message_content = data[0]['message']
    
        # Iterate through the list of messages in the data.
        for message in data:
            # Get the author and message content of the current message.
            current_author = message['user']
            message_content = message['message']
    
            # Check if the previous author is the same as author_a, and the current author is different.
            if prev_author == author_a and current_author != author_a:
                # If the conditions are met, append the previous message as a "message" and the current message as a "response" to the message_pairs list.
                message_pairs.append({"message": prev_message_content,'prompt':'', "response": message_content})
    
            # Update the previous author and previous message content for the next iteration.
            prev_author = current_author
            prev_message_content = message_content
            message_a = message
    
        # Return the list of message pairs.
        return message_pairs
    
    data = []
    input_file = file_path
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Call the find_author_pairs function on the data to find pairs of messages from alternating authors.
    message_pairs = find_author_pairs(data)
    return(message_pairs)

def valid_row(row):
  # word blacklist file is located at ...
  # blacklist = open('...', 'r').read()
  # if row.column NOT in blacklist:                        # TODO
  #   return true
  # else:
    return false

def filter_dataset(file_path):                 # accepts the file path of the un-filtered dataset
  in_file = open(file_path, 'r')
  file_name = os.path.basename(file_path)
  out_file = open('filtered_'+file_name, 'w')
  data = in_file.read()
  for row in data:
    if valid_row(row):                      # sends the row to be validated
      out_file.write(row)
  return('filtered_'+file_path)                # returns the path of the now filtered data set

def convert_to_json(file_path):
    in_file = open(file_path, 'r')
    #print(len(in_file))
    #data = in_file.read()
    #print(len(data))
    out_data = []
    
    for line in in_file:
        if "class=chatlog__author title=" in line:
            #print("row has data")
            user = re.search('data-user-id=\d+', line).group().split("=")[1]
            time = re.search('(?:class=chatlog__timestamp title=")([^"]+)', line).group()
            stripped_time = datetime.strptime(time.split('"')[1], '%A, %d %B %Y %I:%M %p')
            messages = re.finditer('<span class=chatlog__markdown-preserve>[^<]+', line)
            for iter in messages:
                iter_message = iter.group().split(">")[1]
                decoded_string = html.unescape(iter_message)
                out_data.append({
                    "user":user,
                    "time":stripped_time,
                    "message":decoded_string
                })
        else:
            #print(f'no data found in row:\n {line}')
            continue
    #process
    return(out_data)

# model_id = "distilbert-base-uncased"
# api_token = "hf_XXXXXXXX" # get yours at hf.co/settings/tokens

def predict_response(message, prompt, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

#Function to calculate uniqueness score
def calculate_word_uniqueness(word):
    word = nlp(word)
    # word_vector = word.vector
    similarity_scores = [nlp(w).similarity(word) for w in str(nlp.vocab)]
    uniqueness_score = sum(similarity_scores) / len(similarity_scores)
    return uniqueness_score

# Calculate the prompt for one entry
def calculate_prompt(message, response, limit = 1):
    results = []
    
    # Tokenize the messages using spaCy
    message_tokens = nlp(message.lower())
    response_tokens = nlp(response.lower())
    
    # Find prompt words using word subtraction
    prompt_words = list(set([token.text for token in response_tokens]) - set([token.text for token in message_tokens]) - set([token.text for token in response_tokens if "'" in token.text]))

    results = sorted(prompt_words, key=calculate_word_uniqueness, reverse=False)
    #results = sorted(prompt_words, key=calculate_word_uniqueness, reverse=True)
    return(results[:limit])

def sequentialise(input_file_path):
    sequences = []
    
    with open(input_file_path, "r") as input_file:
        input_data = json.load(input_file)
    
    for section in input_data:

        message = section["message"]
        response = section["response"]
        prompt = " ".join(section["prompt"])
    
        sequences.append({'text': f'[MESG]{message}[/MESG][SHRT]{prompt}[/SHRT][LONG]{response}[/LONG]'})
    print("complete")
    return(sequences)

def calculate_prompts(input_file_path):
    # Load a pre-trained spaCy model with word embeddings (e.g., en_core_web_md)
    
    print("loading parameters")
    global nlp
    nlp = spacy.load("en_core_web_md")
    
    #List of stopwords
    stopwords = set(nlp.Defaults.stop_words)
    # Read input data from a JSON file
    
    with open(input_file_path, "r") as input_file:
        input_data = json.load(input_file)
    
    n = len(input_data)
    #n = 20  # Replace 20 with your desired value of n
    num_values = 11
    
    # Generate 10 evenly spaced numbers ranging from 1 to n
    evenly_spaced_numbers = list(np.linspace(1, n, num_values, dtype=int))
    evenly_spaced_numbers = evenly_spaced_numbers[:-1]
    
    working_data = input_data
    
    # Calculate prompt words for each section using word subtraction
    for i, section in enumerate(working_data):
        try:
            print(f"current progress: {evenly_spaced_numbers.index(i)*10:.0f}%")
        except:
            None
            
        message = section["message"]
        response = section["response"]
    
        prompt_words = calculate_prompt(message, response)

        # Store the result for this section in the dictionary
        # input_data[i].prompt = prompt_words
        working_data[i]['prompt'] = prompt_words
    print("complete")
    return (working_data)

def save_json(output_data, output_file_path="uniqueness_scores.json"):
    # Save the results as a JSON file
    with open(output_file_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4, default=str)

    print(f"Results saved to path: {output_file_path}")

def train(dataset_filepath):
    # code taken from https://huggingface.co/blog/falcon#fine-tuning-with-peft
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dataset = load_dataset(dataset_filepath, split="train")

    model_id = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
    )
    trainer.train()
    
    return trainer

def generate_input_texts(training_data):
    # Initialize lists to store combined input text
    input_texts = []
    
    # Combine the input data into input_texts
    for entry in training_data:
        message = entry["message"]
        response_message = entry["response"]
        system_prompt = "You are a helpful assistant. Generate a response to the following message using the prompt."
        prompt = ' '.join(entry["prompt"])
        input_text = f"{system_prompt}\nmessage: {message}\nprompt: {prompt}\nresponse{response_message}\n"
        input_texts.append(input_text)
    return(input_texts)

# Define a custom dataset for training
class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.input_ids.items()}

def truncate(encoded_input):
  encoded_input_trc={}
  for k,v in encoded_input.items():
      v_truncated = v[:,:512]
      encoded_input_trc[k]=v_truncated
  return encoded_input_trc

def split_data(file_path="linked_prompted.json"):
    # Load a pre-trained spaCy model with word embeddings (e.g., en_core_web_md)
    #nlp = spacy.load("en_core_web_md")
    
    # Read input data from a JSON file
    with open(file_path, "r") as input_file:
        input_data = json.load(input_file)
    
    # Shuffle the data randomly to ensure randomness in the split
    random.shuffle(input_data)
    
    # Define the proportions for the split (adjust as needed)
    train_split = 0.7  # 70% for training
    test_split = 0.15  # 15% for testing
    validation_split = 0.15  # 15% for validation
    
    # Calculate the split sizes based on the proportions
    total_samples = len(input_data)
    train_size = int(train_split * total_samples)
    test_size = int(test_split * total_samples)
    
    # Split the data
    train_data = input_data[:train_size]
    test_data = input_data[train_size:train_size + test_size]
    validation_data = input_data[train_size + test_size:]
    
    # Save the split datasets as separate JSON files
    with open("train_data.json", "w") as train_file:
        json.dump(train_data, train_file, indent=4)
    
    with open("test_data.json", "w") as test_file:
        json.dump(test_data, test_file, indent=4)
    
    with open("validation_data.json", "w") as validation_file:
        json.dump(validation_data, validation_file, indent=4)
    
    print("Data split into training, testing, and validation sets.")
    print("created train_data.json, test_data.json, validation_data.json")