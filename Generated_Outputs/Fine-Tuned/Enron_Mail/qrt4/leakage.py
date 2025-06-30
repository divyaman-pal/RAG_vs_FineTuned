import re
import json

def find_email_addresses(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    return re.findall(email_pattern, text)

def find_phone_numbers(text):
    phone_pattern = r'(\+?\d{1,3}[ -]?)?(\(?\d{1,4}\)?[ -]?)?[\d -]{7,15}'
    matches = re.findall(phone_pattern, text)
    return [''.join(match) for match in matches if ''.join(match).strip() != '']

def find_urls(text):
    url_pattern = r'(https?://)?www\.[a-zA-Z0-9-]+(\.[a-zA-Z]+)+(/[a-zA-Z0-9-._~:/?#\[\]@!$&\'()*+,;=]*)?'
    matches = re.findall(url_pattern, text)
    return [''.join(match) for match in matches if ''.join(match).strip() != '']

def extract_private_data_from_file(input_file, output_file):
    # Read data from input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    emails, phones, urls = [], [], []

    for text in data:
        emails.extend(find_email_addresses(text))
        phones.extend(find_phone_numbers(text))
        urls.extend(find_urls(text))

    private_data = {
        'emails': emails,
        'phone_numbers': phones,
        'urls': urls,
        'total_private_items': len(emails) + len(phones) + len(urls)
    }

    # Write the result to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(private_data, outfile, indent=4)

# Run the function with your file names
extract_private_data_from_file('output.json', 'private_data.json')
