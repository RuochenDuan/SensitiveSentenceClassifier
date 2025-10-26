# SensitiveSentenceClassifier
A binary-classifier network trained from X-sensitive dataset. It receives embeddings from MiniLM-v3-v2 and outputs preds which could be sigmoid to a value, measuring whether the sentence is inappropriate.
## Intro
This network depends [`paraphrase-MiniLM-L3-v2`](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2). It is actually a stupid homework finished by a lazy student(me). The accuracy is acceptable(about 72% on the test entries of [`X-sensitive`](https://huggingface.co/datasets/cardiffnlp/x_sensitive)).

- feature: Absolutely fast. Forget the accuracy, just fast.
- structure: layerNorm -> linear -> linear
- scale: 50,177 params, very tiny.

## Quickstart
### Requirements
```bash
pip install -r requirements.txt
```
### Train
Downloading the model and dataset first.
```bash
python train.py
```
### Apply
The weight of `MiniLM-Xsensitive-en-L_class-v3` is provided.
```python
text = "the sentence"
encoder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
model = SensitiveClassifierByMiniLM().to(device)
model.load_state_dict(torch.load(f"{./MiniLM-Xsensitive-en-L_class-v3.pth}"))
model.eval()
with torch.no_grad():
    embedding = encoder.encode(text, convert_to_tensor=True)
    logit = model(embedding.unsqueeze(0))
    prob = torch.sigmoid(logit).item()
    is_sensitive = prob > threshold
```
## License
- **This project's code** is licensed under the **[MIT License](./License)**.
- The `paraphrase-MiniLM-L3-v2` is licensed under **[Apache License 2.0](https://github.com/huggingface/sentence-transformers/blob/master/LICENSE)**. Copyright © 2020 
[Nils Reimers, UKP Lab](https://www.ukp.tu-darmstadt.de/).
- The `X-sensitive` is licensed under **[CC-BY-2.0](https://creativecommons.org/licenses/by/2.0/)**.
Copyright © Cardiff NLP, Cardiff University