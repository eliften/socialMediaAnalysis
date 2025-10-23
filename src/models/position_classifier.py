import numpy as np, torch, yaml, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


class OpinionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length = 256, label2id = None, id2label = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if label2id is not None and id2label is not None:
            self.label2id = label2id
            self.id2label = id2label
            self.label_ids = np.array([label2id[label] for label in labels])
        else:
            self.label_encoder = LabelEncoder()
            self.label_ids = self.label_encoder.fit_transform(labels)
            self.label2id = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_ids[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        
        return {'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)}

class PositionClassifier:  
    def __init__(self, config_path: str = "configs/config.yaml", device: str = None):

        self.config = self._load_config(config_path)
        self.model_config = self.config['models']['position_classifier']
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        self.model_name = self.model_config['model_type']
        self.num_labels = self.model_config['num_labels']
        self.max_length = self.model_config['max_length']
        self.lr = float(self.model_config['learning_rate'])
        self.epochs = self.model_config['epochs']
        self.batch_size = self.model_config['batch_size']
        
        logger.info(f"Loading model: {self.model_name}")
        
        model_mapping = {
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'tiny-bert': 'prajjwal1/bert-tiny'
        }
        
        model_id = model_mapping.get(self.model_name, 'bert-base-uncased')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        logger.info(f"Model loaded with {self.num_labels} labels")
        
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_datasets(self, texts, labels, test_size = 0.2, random_state = 42):

        from sklearn.model_selection import train_test_split
        
        logger.info(f"Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        train_dataset = OpinionDataset(X_train, y_train, self.tokenizer, self.max_length)
        
        self.label2id = train_dataset.label2id
        self.id2label = train_dataset.id2label
        
        val_dataset = OpinionDataset(X_val, y_val, self.tokenizer, self.max_length,
                                     label2id=self.label2id, id2label=self.id2label)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Labels: {self.id2label}")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        logger.info(f"Starting training for {self.epochs} epochs")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_dataset.label_ids),
            y=train_dataset.label_ids
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        for epoch in range(self.epochs):
            print(f"\n Epoch {epoch + 1}/{self.epochs}")
            print("-" * 60)
            
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits, labels)
                
                train_loss += loss.item()
                train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            
            val_loss, val_accuracy = self._evaluate(val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.4f}")
        
        logger.info("Training completed")
    
    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        preds = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                total_loss += outputs.loss.item()
                preds.extend(torch.argmax(outputs.logits, dim=1).detach().cpu().numpy())
                labels.extend(batch_labels.detach().cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(labels, preds)
        
        return avg_loss, accuracy
    
    def predict(self, texts):
        self.model.eval()
        
        dummy_label = list(self.id2label.values())[0]
        dataset = OpinionDataset(texts, [dummy_label] * len(texts), self.tokenizer, self.max_length,
                                label2id=self.label2id, id2label=self.id2label)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                probs = probs.detach().cpu().numpy()
                
                all_preds.extend(preds)
                all_probs.extend(probs)
        
        predicted_labels = [self.id2label[p] for p in all_preds]
        
        return predicted_labels, np.array(all_probs)
    
    def save_model(self, save_path: str):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
    
    def plot_training_history(self, output_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(self.history['train_accuracy'], label='Train Accuracy', marker='o')
        axes[1].plot(self.history['val_accuracy'], label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
        
        plt.show()
        
        return fig


class PositionClassifierEvaluator:   
    @staticmethod
    def calculate_metrics(true_labels,pred_labels, probabilities = None):
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision_weighted': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'precision_macro': precision_score(true_labels, pred_labels, average='macro', zero_division=0),
            'recall_weighted': recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'recall_macro': recall_score(true_labels, pred_labels, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(true_labels, pred_labels,  output_path = None
    ):
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(true_labels)
        y_pred_encoded = le.transform(pred_labels)
        
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le.classes_, yticklabels=le.classes_,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix - Position Classification', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_path}")
        
        plt.show()
        
        return fig
    
    @staticmethod
    def print_classification_report(true_labels, pred_labels):
        report = classification_report(true_labels, pred_labels)
        print("\n" + "="*70)
        print("📊 CLASSIFICATION REPORT")
        print("="*70)
        print(report)