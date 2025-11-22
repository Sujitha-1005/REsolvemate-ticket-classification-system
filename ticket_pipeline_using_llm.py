"""
ticket_pipeline_with_llm.py

- Multi-task classification (category, subcategory, urgency, sentiment)
- Hybrid rule overrides (e.g., refund + invoice -> refund_request)
- LLM-based suggested response template generation + caching (OpenAI)
- CLI: --do_train, --do_infer
"""

import os
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ✅ Modern OpenAI import
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "secret_key_here"
# ---------------------
# Regex / simple rules
# ---------------------
INVOICE_PATTERNS = [r"INV[-_]?[0-9]{3,}", r"invoice\s*#?:?\s*[0-9]{3,}", r"order\s*#?:?\s*[0-9]{3,}"]
REFUND_KEYWORDS = [r"refund", r"money back", r"return", r"reimburse"]

def find_first_pattern(text, patterns):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None

def has_keyword(text, keywords):
    for k in keywords:
        if re.search(k, text, flags=re.IGNORECASE):
            return True
    return False


# ---------------------
# Dataset class
# ---------------------
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['category'] = torch.tensor(self.labels['category'][idx], dtype=torch.long)
        item['subcategory'] = torch.tensor(self.labels['subcategory'][idx], dtype=torch.long)
        item['urgency'] = torch.tensor(self.labels['urgency'][idx], dtype=torch.long)
        item['sentiment'] = torch.tensor(self.labels['sentiment'][idx], dtype=torch.long)
        return item


# ---------------------
# Model
# ---------------------
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, n_category, n_subcategory, n_urgency, n_sentiment, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.cat_head = nn.Linear(hidden, n_category)
        self.sub_head = nn.Linear(hidden, n_subcategory)
        self.urg_head = nn.Linear(hidden, n_urgency)
        self.sent_head = nn.Linear(hidden, n_sentiment)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        pooled = self.drop(pooled)
        return (self.cat_head(pooled),
                self.sub_head(pooled),
                self.urg_head(pooled),
                self.sent_head(pooled))


# ---------------------
# LLM Template generation + caching
# ---------------------
DEFAULT_TEMPLATE_MAP = {
    'refund_request': "Thanks for contacting us. I understand you'd like a refund. Please share your invoice number and we'll investigate and process this as quickly as possible.",
    'payment_failure': "Sorry you're facing payment issues. Could you share the payment method and any error you see? We'll look into it right away.",
    'bug_report': "Thanks for reporting this. Can you please share steps to reproduce, device/browser, and a screenshot if possible?",
    'business_inquiry': "Thanks for your interest. I'll forward this to our partnerships team—someone will reach out within 48 hours.",
    'password_reset': "I can help with a password reset. For security, please confirm your registered email address.",
}

def generate_llm_template_openai(subcategory, sentiment, user_text,
                                 model_name="gpt-4o-mini",
                                 max_tokens=120,
                                 temperature=0.2):
    """Generate a short template using OpenAI LLM."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    system_msg = "You are a concise, empathetic customer support assistant. Produce a short (1-3 sentences) template reply."
    user_prompt = f"Write a short, empathetic customer support reply for subcategory='{subcategory}', sentiment='{sentiment}'. User message: {user_text}\nReply:"

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return resp.choices[0].message.content.strip()


def get_template_for(subcategory, sentiment, user_text, cache_path="template_cache.joblib"):
    key = f"{subcategory}||{sentiment}"
    # load cache
    if os.path.exists(cache_path):
        cache = joblib.load(cache_path)
    else:
        cache = {}

    if key in cache:
        return cache[key], True

    # try LLM if key not in cache
    try:
        template = generate_llm_template_openai(subcategory, sentiment, user_text)
        # save
        cache[key] = template
        joblib.dump(cache, cache_path)
        return template, False
    except Exception as e:
        print("⚠️ OpenAI error, falling back:", e)
        # fallback to default canned template
        template = DEFAULT_TEMPLATE_MAP.get(subcategory, "Thanks for reaching out. We'll route this to the right team and get back to you shortly.")
        cache[key] = template
        joblib.dump(cache, cache_path)
        return template, False


# ---------------------
# Training & evaluation helpers
# ---------------------
def compute_weights(y):
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(w, dtype=torch.float)

def evaluate_model(model, loader, device, label_encoders):
    model.eval()
    all_pred = {'category': [], 'subcategory': [], 'urgency': [], 'sentiment': []}
    all_true = {'category': [], 'subcategory': [], 'urgency': [], 'sentiment': []}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            for i, head in enumerate(['category', 'subcategory', 'urgency', 'sentiment']):
                pred = torch.argmax(logits[i], dim=-1).cpu().numpy()
                all_pred[head].extend(pred.tolist())
                all_true[head].extend(batch[head].cpu().numpy().tolist())
    reports = {}
    for h in all_true:
        reports[h] = {
            'report': classification_report(all_true[h], all_pred[h], target_names=list(label_encoders[h].classes_), zero_division=0),
            'accuracy': accuracy_score(all_true[h], all_pred[h]),
            'f1_macro': f1_score(all_true[h], all_pred[h], average='macro')
        }
    return reports


# ---------------------
# Hybrid inference pipeline (rules + LLM)
# ---------------------
def hybrid_infer(texts, tokenizer, model, device, label_encoders, cache_path="template_cache.joblib"):
    model.eval()
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = [torch.softmax(l, dim=-1).cpu().numpy() for l in logits]

    results = []
    for i, txt in enumerate(texts):
        pred_cat = label_encoders['category'].inverse_transform([int(np.argmax(probs[0][i]))])[0]
        pred_sub = label_encoders['subcategory'].inverse_transform([int(np.argmax(probs[1][i]))])[0]
        pred_urg = label_encoders['urgency'].inverse_transform([int(np.argmax(probs[2][i]))])[0]
        pred_sent = label_encoders['sentiment'].inverse_transform([int(np.argmax(probs[3][i]))])[0]

        overridden = False
        inv = find_first_pattern(txt, INVOICE_PATTERNS)
        if has_keyword(txt, REFUND_KEYWORDS) and inv is not None:
            pred_sub = 'refund_request'
            overridden = True

        # get template (from cache or LLM or fallback)
        try:
            template, from_cache = get_template_for(pred_sub, pred_sent, txt, cache_path=cache_path)
        except Exception:
            template = DEFAULT_TEMPLATE_MAP.get(pred_sub, "Thanks for contacting us. We'll reply shortly.")
            from_cache = False

        results.append({
            'text': txt,
            'predicted_category': pred_cat,
            'predicted_subcategory': pred_sub,
            'predicted_urgency': pred_urg,
            'predicted_sentiment': pred_sent,
            'overridden_by_rule': overridden,
            'invoice_detected': inv,
            'suggested_template': template,
            'template_from_cache': from_cache
        })
    return results


# ---------------------
# Main CLI
# ---------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.csv)
    df['customer_message'] = df['customer_message'].astype(str)

    # label encoding
    le = {}
    for col in ['category', 'subcategory', 'urgency', 'sentiment']:
        le[col] = LabelEncoder()
        df[col + '_enc'] = le[col].fit_transform(df[col])

    label_encoders = {k: le[k] for k in le}

    # split
    train_df, val_df = train_test_split(df, test_size=args.val_size, stratify=df['category'], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_labels = {
        'category': train_df['category_enc'].values,
        'subcategory': train_df['subcategory_enc'].values,
        'urgency': train_df['urgency_enc'].values,
        'sentiment': train_df['sentiment_enc'].values
    }
    val_labels = {
        'category': val_df['category_enc'].values,
        'subcategory': val_df['subcategory_enc'].values,
        'urgency': val_df['urgency_enc'].values,
        'sentiment': val_df['sentiment_enc'].values
    }

    if args.do_train:
        train_texts = train_df['customer_message'].tolist()
        val_texts = val_df['customer_message'].tolist()

        train_ds = TicketDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
        val_ds = TicketDataset(val_texts, val_labels, tokenizer, max_len=args.max_len)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = MultiTaskModel(args.model_name,
                               n_category=len(le['category'].classes_),
                               n_subcategory=len(le['subcategory'].classes_),
                               n_urgency=len(le['urgency'].classes_),
                               n_sentiment=len(le['sentiment'].classes_),
                               dropout=args.dropout)
        model.to(device)

        urg_w = compute_weights(train_labels['urgency']).to(device)
        sent_w = compute_weights(train_labels['sentiment']).to(device)

        crit = {
            'category': nn.CrossEntropyLoss(),
            'subcategory': nn.CrossEntropyLoss(),
            'urgency': nn.CrossEntropyLoss(weight=urg_w),
            'sentiment': nn.CrossEntropyLoss(weight=sent_w)
        }

        optimizer = AdamW([
            {'params': model.encoder.parameters(), 'lr': args.lr},
            {'params': model.cat_head.parameters(), 'lr': args.lr},
            {'params': model.sub_head.parameters(), 'lr': args.lr},
            {'params': model.urg_head.parameters(), 'lr': args.head_lr},
            {'params': model.sent_head.parameters(), 'lr': args.head_lr},
        ], lr=args.lr)

        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

        best_f1 = 0.0
        os.makedirs(args.output_dir, exist_ok=True)
        joblib.dump(label_encoders, os.path.join(args.output_dir, 'label_encoders.joblib'))

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                lab_cat = batch['category'].to(device)
                lab_sub = batch['subcategory'].to(device)
                lab_urg = batch['urgency'].to(device)
                lab_sent = batch['sentiment'].to(device)

                logits_cat, logits_sub, logits_urg, logits_sent = model(input_ids=input_ids, attention_mask=attention_mask)

                loss_cat = crit['category'](logits_cat, lab_cat)
                loss_sub = crit['subcategory'](logits_sub, lab_sub)
                loss_urg = crit['urgency'](logits_urg, lab_urg)
                loss_sent = crit['sentiment'](logits_sent, lab_sent)

                loss = loss_cat + loss_sub + loss_urg + loss_sent
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{args.epochs} Train loss: {total_loss/len(train_loader):.4f}")
            reports = evaluate_model(model, val_loader, device, label_encoders)
            for head, info in reports.items():
                print(f"--- {head} ---")
                print(info['report'])
                print(f"Accuracy: {info['accuracy']:.4f} F1_macro: {info['f1_macro']:.4f}")

            avg_f1 = np.mean([reports[h]['f1_macro'] for h in reports])
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
                tokenizer.save_pretrained(args.output_dir)
                print("Saved best model.")

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
        tokenizer.save_pretrained(args.output_dir)
        print("Training complete. Artifacts in", args.output_dir)

    if args.do_infer:
        model = MultiTaskModel(args.model_name,
                               n_category=len(le['category'].classes_),
                               n_subcategory=len(le['subcategory'].classes_),
                               n_urgency=len(le['urgency'].classes_),
                               n_sentiment=len(le['sentiment'].classes_),
                               dropout=args.dropout)
        model_path = os.path.join(args.model_dir if args.model_dir else args.output_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found. Train first or point --model_dir to folder with best_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_dir else args.output_dir, use_fast=True)
        label_encoders = joblib.load(os.path.join(args.model_dir if args.model_dir else args.output_dir, 'label_encoders.joblib'))

        texts = [args.infer_text] if args.infer_text else list(df['customer_message'].astype(str).values[:10])
        results = hybrid_infer(texts, tokenizer, model, device, label_encoders, cache_path=args.template_cache)
        for r in results:
            print("="*60)
            print("TEXT:", r['text'])
            print("Category:", r['predicted_category'])
            print("Subcategory:", r['predicted_subcategory'], "(overridden:", r['overridden_by_rule'], ")")
            print("Urgency:", r['predicted_urgency'])
            print("Sentiment:", r['predicted_sentiment'])
            print("Invoice found:", r['invoice_detected'])
            print("Template (cached?):", r['template_from_cache'])
            print("Suggested template:\n", r['suggested_template'])
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="customer_support_dataset.csv")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.12)
    parser.add_argument("--output_dir", type=str, default="ticket_model")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_infer", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--infer_text", type=str, default=None)
    parser.add_argument("--template_cache", type=str, default="template_cache.joblib")
    args = parser.parse_args()
    main(args)
