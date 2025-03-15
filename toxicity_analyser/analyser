import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from transformers import *
from torch.autograd import Variable
import io

# Pretrained BERT model for embeddings
from transformers import BertModel

# Load the pretrained BERT model for embeddings
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()  # Set BERT to evaluation mode


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

model_path = r"C:\Users\HP\OneDrive\Documents\Desktop\TextGenie\TextGenie\toxicity_analyzer\toxicDetection_model.pkl"
# Load the tokenizer (assuming the model is based on BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

#Yoon Kim CNN Stuff
class kim_cnn(nn.Module):
  def __init__(self, embedding_num, embedding_dim, dropout=0.1, kernel_count=3, kernel_dims = [2, 3, 4], labels = labels ):
    super().__init__()
    self.dropout = dropout
    self.kernel_count = kernel_count
    self.kernel_dims = kernel_dims
    self.labels = labels
    self.label_count = len(labels)
    self.emb_num = embedding_num
    self.emb_dim = embedding_dim

    self.embedding = nn.Embedding(self.emb_num, self.emb_dim)
    self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_count, (k, self.emb_dim)) for k in self.kernel_dims])
    self.dropout = nn.Dropout(self.dropout)
    self.classifier = nn.Linear(len(self.kernel_dims) * self.kernel_count, len(self.labels))
    self.act = nn.Sigmoid()
    # self.act = nn.ReLU()
    # self.act = nn.Softmax(dim = 1)

  def forward(self, X):
    # (N, W, D) ---> (N, C, W, D)
    X = Variable(X, requires_grad = True)
    X = X.unsqueeze(1)
    # [(N, C, W), ...] * len(kernel_dims)
    X = [F.relu(conv(X)).squeeze(3) for conv in self.convs]
    # concat([(N, C), ...] * len(kernel_dims))
    X = [F.max_pool1d(n, n.size(2)).squeeze(2) for n in X]
    X = torch.cat(X, 1)
    # (N, len(kernel_dims) * kernel_count)
    X = self.dropout(X)
    #(N, C)
    logits = self.act(self.classifier(X))
    return logits


# Load the model
def load_model(model_path):
    device = torch.device('cpu')
    # model = pickle.load(f, fix_imports=True, encoding='latin1')
    # model = kim_cnn(
    #     embedding_num=256,  # like the sequence_length value we set last time?
    #     embedding_dim=768,  # x_train[0].shape[2], #BERT embeedings
    #     dropout=0.1,
    #     kernel_count=3,
    #     kernel_dims=[2, 3, 4],
    #     labels=labels
    # )
    with open(model_path, 'rb') as f:

        model =CPU_Unpickler(f).load()
        # model = torch.load(f,map_location = torch.device('cpu'))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def predict_text_toxicity(text_input):
    # Preprocess the input text
    inputs = tokenizer(text_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']  # Token IDs
    attention_mask = inputs['attention_mask']  # Attention mask

    # Generate embeddings using BERT
    with torch.no_grad():
        embeddings = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state

    # Load the model
    model = load_model(model_path)

    # Pass embeddings to the model
    output = model(embeddings)

    # Get probabilities
    prob = output.squeeze().tolist()

    return prob, labels
