import os
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def get_predictions(text):
    url = "http://127.0.0.1:8000/predict"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    return response.json()

def main():
    # Prompt the user to enter the file path
    file_path = '/Users/Srimouli/Documents/experiments/Flash/ML Engineer - Assignment/Flash.co ML Engineering - Assignment Training data.xlsx'  # input("Enter the file path: ")

    # Check the file extension and read the data accordingly
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.csv':
        data = pd.read_csv(file_path)
    elif file_extension.lower() == '.xlsx':
        data = pd.read_excel(file_path)
    else:
        print("Invalid file type. Only CSV and Excel files are supported.")
        return

    # Assuming your data has columns 'text', 'expected_category', and 'expected_email_type'
    text_column = 'Text'
    expected_category_column = 'Category'
    expected_email_type_column = 'EmailType'

    true_categories = []
    pred_categories = []
    true_email_types = []
    pred_email_types = []

    # Use tqdm to display a progress bar
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        text = row[text_column]
        expected_category = row[expected_category_column]
        expected_email_type = row[expected_email_type_column]

        predictions = get_predictions(text)
        pred_category = predictions['category']
        pred_email_type = predictions['email_type']

        true_categories.append(expected_category)
        pred_categories.append(pred_category)
        true_email_types.append(expected_email_type)
        pred_email_types.append(pred_email_type)

    category_accuracy = accuracy_score(true_categories, pred_categories)
    email_type_accuracy = accuracy_score(true_email_types, pred_email_types)

    print(f"Category accuracy: {category_accuracy}")
    print(f"Email type accuracy: {email_type_accuracy}")

if __name__ == "__main__":
    main()