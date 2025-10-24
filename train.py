#!usr/bin/env python3
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import OneCycleLR
from classifier import SensitiveClassifierByMiniLM


model_name = "./paraphrase-MiniLM-L3-v2"
batch_size = 16
max_epochs = 20
patience = 3
lr = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "./MiniLM-Xsensitive-en-L_class-v3.pth"


def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            sentences.append(data["text"])
            labels.append(1 if data.get("#labels", 0) > 0 else 0)
    return sentences, labels


def encode_sentences(encoder, sentences):
    return encoder.encode(sentences, convert_to_tensor=True, device=device)


def create_data_loader(encoder, sentences, labels, batch_size, shuffle=False):
    embeddings = encode_sentences(encoder, sentences)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(embeddings, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(train_loader, dev_loader, model, criterion, optimizer, scheduler, max_epochs, patience, save_path, device):
    best_f1 = -1.0
    counter = 0
    early_stop = False

    for epoch in range(max_epochs):
        if early_stop:
            break
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for inputs, labels in tqdm(dev_loader, desc=f"Epoch {epoch + 1} validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_true, all_preds, average='binary')
        print(f"Validation F1 Score: {val_f1:.4f}")
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best F1: {best_f1:.4f}. Model saved.")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")
            if counter >= patience:
                early_stop = True


if __name__ == '__main__':
    print("Initializing...")
    encoder = SentenceTransformer(model_name).eval()

    train_sentences, train_labels = load_data("./Xsensitive/train.jsonl")
    dev_sentences, dev_labels = load_data("./Xsensitive/validation.jsonl")
    test_sentences, test_labels = load_data("./Xsensitive/test.jsonl")
    train_loader = create_data_loader(encoder, train_sentences, train_labels, batch_size, shuffle=True)
    dev_loader = create_data_loader(encoder, dev_sentences, dev_labels, batch_size, shuffle=False)
    test_loader = create_data_loader(encoder, test_sentences, test_labels, batch_size, shuffle=False)

    model = SensitiveClassifierByMiniLM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=max_epochs)

    print("Training...")
    train(
        train_loader=train_loader,
        dev_loader=dev_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        patience=patience,
        save_path=save_path,
        device=device
    )

    print("Evaluating...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="binary")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
