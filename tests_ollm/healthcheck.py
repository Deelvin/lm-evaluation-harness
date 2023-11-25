import os
import json
import argparse

import requests
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)


def fail_message(msg):
    print(Fore.RED + "✗", end=' ')  
    print(f"{msg}") 
    
def success_message(msg):      
    print(Fore.GREEN + "✓", end=' ')  # Print checkmark symbol in green
    print(f"{msg}")

def construct_request_url(endpoint, prod):
    return f"{endpoint['url']}/v1/chat/completions"
    
def check_endpoint_health(endpoint, token, prod=True):
    
    print(f'Model: {endpoint["model"]}')

    try:
        request_url = construct_request_url(endpoint, prod)
        print(request_url)    
        response = requests.post(request_url, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}, json={
            "model": endpoint['model'],
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": "Who is Python author?"
                }
            ],
            "max_tokens": 100
        }, timeout=10)  # Set timeout to 5 seconds

        if response.status_code == 200:
            success_message(f"Basic request. {response.status_code}")
        else:
            fail_message(f"Basic request. {response.status_code}")
    except requests.RequestException:
        fail_message("Basic request")

def check_all_endpoints(endpoints, token):
    for endpoint in endpoints:
        check_endpoint_health(endpoint, token)
        print()  # Add a new line for better readability between endpoint reports

def parse_arguments():
    parser = argparse.ArgumentParser(description='Check endpoint health')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Fetch OCTOAI_TOKENs from environment variable
    octoai_token_dev = os.getenv('OCTOAI_TOKEN_DEV')
    octoai_token_prod = os.getenv('OCTOAI_TOKEN_PROD')

    if (args.prod == True):
        if octoai_token_prod is None:
            print(Fore.RED + "OCTOAI_TOKEN_PROD environment variable not found.")
            exit(1)
        else:
            octoai_token = octoai_token_prod
    else:
        if octoai_token_dev is None:
            print(Fore.RED + "OCTOAI_TOKEN_DEV environment variable not found.")
            exit(1)
        else:
            octoai_token = octoai_token_dev
    
    # Load endpoint details from JSON file
    with open('endpoints.json', 'r') as file:
        data = json.load(file)
        if (args.prod):
            endpoints = data.get('prod', [])
        else:
            endpoints = data.get('dev', [])
            
    # Check health for all endpoints
    check_all_endpoints(endpoints, octoai_token, args.prod)
