import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, AutoModel
from torch.optim import AdamW

print("Loading data...")
real = pd.read_csv('./Fake_Real_Data_Set/True.csv')
unreal = pd.read_csv('./Fake_Real_Data_Set/Fake.csv')

def clean_txt(txt):
    if not isinstance(txt, str):
        return ""
    
    # news source ending with a "- "
    txt = re.sub(r'^.*?\(Reuters\) - ', '', txt)
    
    return txt.strip()

real['text'] = real['text'].apply(clean_txt)
unreal['text'] = unreal['text'].apply(clean_txt)

##not using the title of the data sets to train off of
'''real['title'] = real['title'].apply(clean_txt)
unreal['title'] = unreal['title'].apply(clean_txt)'''

real['lab'] = 0  
unreal['lab'] = 1  

df = pd.concat([real, unreal])
df = df.sample(frac=1).reset_index(drop=True)  

x_train, x_test, y_train, y_test = train_test_split(
    df['text'], 
    df['lab'], 
    test_size=0.2, 
    random_state=42
)

print("Loading BERT...")
tkzr = BertTokenizerFast.from_pretrained('bert-base-uncased')
base_model = AutoModel.from_pretrained('bert-base-uncased')

for p in base_model.parameters():
    p.requires_grad = False

max_len = 20
train_tok = tkzr(x_train.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
test_tok = tkzr(x_test.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="pt")

class NewsModel(nn.Module):
    def __init__(self):
        super(NewsModel, self).__init__()
        self.bert = base_model
        self.clf = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.clf(out['pooler_output'])

train_ds = torch.utils.data.TensorDataset(
    train_tok['input_ids'],
    train_tok['attention_mask'],
    torch.tensor(y_train.values, dtype=torch.long)
)
test_ds = torch.utils.data.TensorDataset(
    test_tok['input_ids'],
    test_tok['attention_mask'],
    torch.tensor(y_test.values, dtype=torch.long)
)

b_size = 15
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=b_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=b_size)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewsModel().to(dev)
loss_fn = nn.CrossEntropyLoss()
opt = AdamW(model.parameters(), lr=0.001)

print(f"Training on {dev}...")
n_epochs = 3

for epoch in range(n_epochs):
    model.train()
    total = 0
    
    for batch in train_dl:
        ids, mask, labs = [b.to(dev) for b in batch]
        
        opt.zero_grad()
        outputs = model(ids, mask)
        loss = loss_fn(outputs, labs)
        
        loss.backward()
        opt.step()
        
        total += loss.item()
    
    avg = total / len(train_dl)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg:.4f}")

model.eval()
preds = []
actuals = []

with torch.no_grad():
    for batch in test_dl:
        ids, mask, labs = [b.to(dev) for b in batch]
        outputs = model(ids, mask)
        _, pred = torch.max(outputs, 1)
        
        preds.extend(pred.cpu().numpy())
        actuals.extend(labs.cpu().numpy())

print("\nTest Results:")
print(classification_report(actuals, preds, target_names=['True', 'Fake']))

cm = confusion_matrix(actuals, preds)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], ['True', 'Fake'])
plt.yticks([0, 1], ['True', 'Fake'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.show()

examples = [
    "Government Passes New Healthcare Bill",
    "Aliens Found Living Among Us, Says Anonymous Source",
    "Stock Market Reaches Record High",
    "Scientists Discover Cure for Cancer in Common Household Item"
]

ex_tok = tkzr(examples, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
ex_ids = ex_tok['input_ids'].to(dev)
ex_mask = ex_tok['attention_mask'].to(dev)

model.eval()
with torch.no_grad():
    out = model(ex_ids, ex_mask)
    _, p = torch.max(out, 1)

print("\nExample Headlines:")
for headline, pred in zip(examples, p):
    result = "Fake News" if pred == 1 else "Real News"
    print(f"{headline} -> {result}")