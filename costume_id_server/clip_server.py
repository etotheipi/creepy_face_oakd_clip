import io
import json                    
import base64                  
import logging             
import numpy as np
from PIL import Image
from datetime import datetime

from flask import Flask, request, jsonify, abort

import torch
import clip
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using cpu or cuda:', device)
model, preprocess = clip.load("ViT-B/32", device=device)

global costume_list
global candidate_texts
global candidate_text_embeds
costume_list = []
candidate_texts = []
candidate_text_embeds = []

def reload_costume_list():
    global costume_list
    global candidate_texts
    global candidate_text_embeds
    costume_list = open('costumes.txt', 'r').read().split('\n')
    costume_list = [c.strip() for c in costume_list if len(c.strip())>2]
    print('All costumes:', costume_list)

    candidate_texts = [f'A kid wearing a {v} costume' for v in costume_list]

    candidate_text_toks = clip.tokenize(candidate_texts).to(device)
    with torch.no_grad():
        candidate_text_embeds = model.encode_text(candidate_text_toks)

    print(candidate_text_embeds.cpu().numpy()) 
    
    
def generate_results_image(orig_img_obj, texts, scores):
    print(type(orig_img_obj))
    if isinstance(orig_img_obj, str):
        orig_img_obj = np.asarray(Image.open(orig_img_obj))
        
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(orig_img_obj)
    ax[0].axis('off')
    
    N = len(texts)
    ax[1].barh(np.arange(N), scores, align='center')
    ax[1].set_yticks(np.arange(N))
    ax[1].set_yticklabels(texts)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_xlabel('Score')
    ax[1].set_title('Classification Score')
    ax[1].set_xlim([-0.05, 1.05])
    
    fig.tight_layout(pad=5)
    
    # Save result image to file
    curr_time_str = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    save_fn = os.path.join('classifier_results', f'results_{curr_time_str}.png')
    fig.savefig(save_fn, dpi=72, facecolor='white')
    
    # Save original image to file (for later experimentation)
    save_fn = os.path.join('classifier_results', 'orig_images', f'img_{curr_time_str}.png')
    #im = Image.fromarray(orig_img_obj)
    im = orig_img_obj
    im.save(save_fn)
    
    # Save image to base64 string to pass back to caller
    img_out = io.BytesIO()
    fig.savefig(img_out, dpi=72)
    img_b64 = base64.b64encode(img_out.getvalue())
    return img_b64.decode('utf8')
    

def identify_trick_or_treater(pil_img, print_results=True):
    global costume_list
    global candidate_texts
    global candidate_text_embeds
    
    image = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        
    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    candidate_text_embeds /= candidate_text_embeds.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ candidate_text_embeds.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    print(type(values), type(indices))
    values = values.cpu().numpy().tolist()
    indices = indices.cpu().numpy().tolist()

    # Print the result
    if print_results:
        print("-"*80)
        print("Top predictions:\n")
        for value, index in zip(values, indices):
            print(f"{candidate_texts[index]:>16s}: {100 * value:.2f}%")
            
    text_labels = [candidate_texts[i].split(' ')[4:-1] for i in indices]
    img_b64 = generate_results_image(pil_img, text_labels, values)
        
    return {
        'scores': values,
        'text_indices': indices,
        'image_base64_utf8': img_b64
    }

  
@app.route("/reload", methods=['GET'])
def reload():         
    reload_costume_list()
    return {
        'status': "SUCCESS",
        'costume_list': [c.split(' ')[-2] for c in candidate_texts]
    }
    
    
@app.route("/classify", methods=['POST'])
def classify():         
    global costume_list
    global candidate_texts
    global candidate_text_embeds
    with open('request_dump.txt', 'w') as f:
        f.write('-'*80)
        f.write(json.dumps(request.json))
        f.write('\n\n')
        
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    res = identify_trick_or_treater(img)
    #print(json.dumps(res))
     
    result_dict = {
        'scores': res['scores'],
        'texts': [candidate_texts[i] for i in res['text_indices']],
        'image_base64_utf8': res['image_base64_utf8']
    }
    return result_dict
  
  
def run_server_api():
    app.run(host='0.0.0.0', port=8080)
  
  
if __name__ == "__main__":     
    reload_costume_list()
    run_server_api()
    
