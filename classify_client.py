

import base64
import json                    
import argparse                    

import requests

# Costume ID server is on the local network.  
API = 'http://192.168.0.239:8080/classify'

def submit(image_file, post_url=API):
    """
    Submits the image as a base64-encoded string to the server via REST/HTTP POST
    """
    with open(image_file, "rb") as f:
        im_bytes = f.read()        
        
    # Base64-encode the image to send over REST/HTTP POST
    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64})
    response = requests.post(post_url, data=payload, headers=headers)
    print(response)
    try:
        data = response.json()     
        print(data)                
        return data
    except requests.exceptions.RequestException:
        print(response.text)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Submit trick-or-treater image for classification',
                    description = 'Classify using zero-shot CLIP')
    
    parser.add_argument('image_fn', type=str, help='Image file to submit')

    args = parser.parse_args()
    print(args.image_fn)
    submit(args.image_fn)
                    
