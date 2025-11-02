import requests

BOT_TOKEN = "8028448884:AAEX48_FMvxUY3ZmEOpcXH9dvYVDYHiH8CE"
url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

response = requests.get(url)
data = response.json()

if data['result']:
    for update in data['result']:
        chat_id = update['message']['chat']['id']
        username = update['message']['chat'].get('username', 'No username')
        print(f"Chat ID: {chat_id}")
        print(f"Username: @{username}")
else:
    print("No messages found. Send a message to your bot first!")