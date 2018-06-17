import requests
import numpy as np


def notify(*messages, token="o0Xs2ymLhJulwT9v9f4thY4ASKVJ3J0FV9M0o4AtZJ3"):
    """
    LINEにmessageを通知
    """
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": "Bearer " + token}

    message = '\n'.join(messages)

    payload = {"message":  message}

    r = requests.post(url,headers=headers,params=payload)
    print(r)