from django.shortcuts import render
from django.http import HttpResponse
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
# from torchModel import BERTGRUSentiment
from nlpapp.torchModel import BERTGRUSentiment
# Create your views here.

from django.views.decorators.csrf import csrf_exempt


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


@csrf_exempt
def sentimentAnalysis(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    userText = body['text']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    device = torch.device('cpu')

    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    bert = BertModel.from_pretrained('bert-base-uncased')

    model = BERTGRUSentiment(bert,
                             HIDDEN_DIM,
                             OUTPUT_DIM,
                             N_LAYERS,
                             BIDIRECTIONAL,
                             DROPOUT)

    model.load_state_dict(torch.load(
        './nlpapp/tut6-model.pt', map_location=torch.device('cpu')))

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    return HttpResponse(predict_sentiment(model, tokenizer, device, userText))


def predict_sentiment(model, tokenizer, device, sentence):
    # model.load_state_dict(torch.load('tut6-model.pt'))

    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id

    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + \
        tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))

    return prediction.item()
