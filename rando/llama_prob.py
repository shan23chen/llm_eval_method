import argparse
import pandas as pd
import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

# Define the argument parser
parser = argparse.ArgumentParser(description='Run inference on a dataset with LLaMA and generate a classification report.')
parser.add_argument('--csv_path', type=str, default='/home/shan/Desktop/netlab/Esophagitis/Proper_split_data/dev.csv', help='Path to the CSV file containing the data.')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='Model name or path.')
parser.add_argument('--num_labels', type=int, default=4, help='Number of labels for the classification task.')
parser.add_argument('--prompt', type=str, default='', help='Optional prompt to prepend to each text input before inference.')

# Parse arguments
args = parser.parse_args()

# Function to perform inference and log results
def perform_inference_and_log_results(dataframe, model_name, num_labels, prompt=''):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.eval()  # Set the model to evaluation mode

    # Containers for true labels and predictions
    true_labels = []
    predictions = []
    logits_list = []

    # Perform inference with progress bar
    for text, true_label in tqdm(zip(dataframe['Full Text'], dataframe['LABEL']), total=len(dataframe)):
        # Prepend prompt if provided
        text = prompt + text if prompt else text
        
        # Tokenize and encode the text
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        # Perform the inference
        with torch.no_grad():  # Ensure no gradients are calculated
            output = model(**encoded_input)
        
        # Extract logits and the predicted label
        logits = output.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        
        # Append to lists
        true_labels.append(true_label)
        predictions.append(predicted_label)
        logits_list.append(logits.detach().cpu().numpy().tolist())

    # Calculate and print classification report
    true_labels = [str(i) for i in true_labels]
    predictions = [str(i) for i in predictions]
    report = classification_report(true_labels, predictions, target_names=model.config.id2label.values())
    print(report)

    return true_labels, predictions, logits_list, report

def main():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(args.csv_path)
    df['LABEL'] = ['Grade ' + str(i) for i in df['Free Text Grade']]

    # Perform inference and log results
    true_labels, predictions, logits_list, report = perform_inference_and_log_results(df, args.model_name, args.num_labels, args.prompt)

    # Optional: Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(inference_results)
    results_df.to_csv('eso_inference_results.csv', index=False)

if __name__ == '__main__':
    main()


