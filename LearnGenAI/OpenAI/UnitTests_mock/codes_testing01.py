import requests

def charge_card(card_number, amount):
    # Simulate an API call to a payment gateway
    response = requests.post("https://api.paymentgateway.com/charge", json={
        "card_number": card_number,
        "amount": amount
    })
    return response.json()