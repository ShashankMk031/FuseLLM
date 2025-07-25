"""
NLP Processor Module

Handles natural language understanding including intent classification and entity recognition.
Uses a pre-trained transformer model for accurate intent detection.
"""
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import re
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    Pipeline,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import logging
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for model evaluation.
    
    Args:
        pred: Model predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class NLPProcessor:
    """
    Handles NLP processing tasks including intent classification and entity recognition.
    
    Uses a pre-trained transformer model for accurate intent detection and can be
    extended with custom training data.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the NLP processor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        
        # Define supported intents and their labels with more examples
        self.intents = {
            "greeting": {
                "examples": [
                    "hi", "hello", "hey", "hi there", "hello there", 
                    "good morning", "good afternoon", "good evening",
                    "greetings", "howdy", "what's up"
                ],
                "description": "General greetings and salutations"
            },
            "joke": {
                "examples": [
                    "tell me a joke", "make me laugh", "funny story",
                    "do you know any jokes?", "tell me something funny",
                    "joke please", "make me smile", "entertain me"
                ],
                "description": "Requests for jokes or humorous content"
            },
            "weather": {
                "examples": [
                    "what's the weather", "will it rain", "temperature today",
                    "is it going to rain today?", "what's the weather forecast",
                    "do I need an umbrella", "how hot is it outside",
                    "weather report"
                ],
                "description": "Weather-related queries"
            },
            "definition": {
                "examples": [
                    "what is a laptop", "define computer", "meaning of artificial intelligence",
                    "explain quantum computing", "what does CPU stand for",
                    "definition of machine learning", "what is the meaning of life",
                    "explain like I'm five", "simple explanation of"
                ],
                "description": "Requests for definitions or explanations"
            },
            "science_question": {
                "examples": [
                    "Newton's laws of motion", "explain photosynthesis",
                    "what is the theory of relativity", "how do black holes work",
                    "explain quantum mechanics", "what is DNA",
                    "how does the brain work", "what is the periodic table"
                ],
                "description": "Scientific questions and explanations"
            },
            "general_knowledge": {
                "examples": [
                    "who is the president", "capital of france",
                    "how to make coffee", "what time is it in london",
                    "history of rome", "how to tie a tie",
                    "best way to learn programming", "how to cook pasta"
                ],
                "description": "General knowledge questions"
            },
            "goodbye": {
                "examples": [
                    "bye", "see you", "goodbye", "take care",
                    "see you later", "bye for now", "talk to you later",
                    "have a good one", "take it easy"
                ],
                "description": "Farewell messages"
            },
            "help": {
                "examples": [
                    "help", "what can you do", "how does this work",
                    "can you help me", "what are my options",
                    "i need assistance", "can you guide me",
                    "what help can you provide"
                ],
                "description": "Requests for help or information about capabilities"
            }
        }
        
        # Define model save path
        self.model_dir = './models/intent_classifier'
        
        # Check if fine-tuned model exists
        if os.path.exists(self.model_dir):
            try:
                logger.info("Loading pre-trained model...")
                # First load the config to get label mappings
                config = AutoConfig.from_pretrained(self.model_dir)
                
                # If config has id2label, use it, otherwise create from intents
                if hasattr(config, 'id2label') and config.id2label:
                    self.id2label = {int(k): v for k, v in config.id2label.items()}
                    self.label2id = {v: int(k) for k, v in config.id2label.items()}
                else:
                    # Fallback to default label mapping
                    self.label2id = {label: i for i, label in enumerate(self.intents.keys())}
                    self.id2label = {i: label for i, label in enumerate(self.intents.keys())}
                
                # Now load the tokenizer and model with the correct config
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir,
                    id2label=self.id2label,
                    label2id=self.label2id,
                    ignore_mismatched_sizes=True
                )
                
                if self.device != -1:
                    self.model = self.model.to(f"cuda:{self.device}")
                logger.info("Pre-trained model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading pre-trained model: {e}")
                self._initialize_and_train()
        else:
            self._initialize_and_train()
            
        # Create a text classification pipeline with updated parameters
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            top_k=None,  # Return all scores
            function_to_apply='sigmoid'  # For multi-label classification if needed
        )
        
        logger.info(f"NLP Processor initialized with model: {model_name}")
    
    def _prepare_training_data(self):
        """Prepare training data from intent examples."""
        self.train_texts = []
        self.train_labels = []
        
        # Create balanced training data
        max_examples = max(len(intent_data["examples"]) for intent_data in self.intents.values())
        
        for label, intent_data in self.intents.items():
            examples = intent_data["examples"]
            # Oversample to balance the classes
            examples = examples * (max_examples // len(examples) + 1)
            examples = examples[:max_examples]
            
            self.train_texts.extend(examples)
            self.train_labels.extend([self.label2id[label]] * len(examples))
        
        # Shuffle the data
        combined = list(zip(self.train_texts, self.train_labels))
        random.shuffle(combined)
        self.train_texts, self.train_labels = zip(*combined)
        
        # Create a dataset
        self.train_dataset = Dataset.from_dict({
            'text': list(self.train_texts),
            'label': list(self.train_labels)
        })
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
            
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    def _fine_tune(self, num_epochs: int = 5, batch_size: int = 8):
        """
        Fine-tune the model on the intent examples.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        try:
            # Set up training arguments compatible with current transformers version
            training_args = TrainingArguments(
                output_dir='./models/intent_classifier',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_ratio=0.1,
                weight_decay=0.01,
                learning_rate=2e-5,
                logging_dir='./logs',
                logging_steps=5,
                save_steps=100,
                save_total_limit=1,
                no_cuda=self.device == -1,
                disable_tqdm=True
            )
            
            # Use the full dataset for training (no eval split in this version)
            train_dataset = self.train_dataset
            
            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                compute_metrics=compute_metrics,
            )
            
            # Train the model
            logger.info("Fine-tuning the model on intent examples...")
            trainer.train()
            
            # Save the fine-tuned model and tokenizer
            trainer.save_model(training_args.output_dir)
            self.tokenizer.save_pretrained(training_args.output_dir)
            
            # Reload the model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(training_args.output_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
            
            # Move model to GPU if available
            if self.device != -1:
                self.model = self.model.to(f"cuda:{self.device}")
            
            logger.info(f"Model fine-tuning completed and saved to {training_args.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def _initialize_and_train(self):
        """Initialize the model, prepare data, and fine-tune."""
        try:
            # Initialize the model and tokenizer
            self._initialize_model()
            
            # Generate training data from examples
            self._prepare_training_data()
            
            # Fine-tune the model
            self._fine_tune()
            
        except Exception as e:
            logger.error(f"Error during model initialization and training: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            # Initialize tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set up label mappings
            self.label2id = {label: i for i, label in enumerate(self.intents.keys())}
            self.id2label = {i: label for i, label in enumerate(self.intents.keys())}
            
            # Initialize the model with the correct number of labels
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True  # In case the pretrained model has a different number of labels
            )
            
            # Move model to GPU if available
            if self.device != -1:
                self.model = self.model.to(f"cuda:{self.device}")
            
            logger.info(f"Model initialized with {len(self.label2id)} intents")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def detect_intents(self, text: str, threshold: float = 0.3) -> List[Dict[str, float]]:
        """
        Detect intents from the input text.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for including intents in results
            
        Returns:
            List of dictionaries containing intent and confidence score
        """
        if not text.strip():
            return []
            
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            # Move inputs to the same device as the model
            if self.device != -1:
                inputs = {k: v.to(f"cuda:{self.device}") for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to list of (label, score) tuples
            results = []
            for i, score in enumerate(probs[0]):
                label = self.id2label[i]
                score = float(score)
                if score >= threshold:
                    results.append({
                        'intent': label,
                        'score': score,
                        'description': self.intents.get(label, {}).get('description', '')
                    })
            
            # Sort by confidence score (highest first)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # If no intent meets the threshold, return the top intent
            if not results and probs.numel() > 0:
                top_idx = torch.argmax(probs[0]).item()
                top_label = self.id2label[top_idx]
                top_score = float(probs[0][top_idx])
                results.append({
                    'intent': top_label,
                    'score': top_score,
                    'description': self.intents.get(top_label, {}).get('description', '')
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in intent detection: {str(e)}")
            # Fallback to a simple keyword-based approach if model fails
            return self._fallback_intent_detection(text)
    
    def _fallback_intent_detection(self, text: str) -> List[Dict[str, float]]:
        """
        Fallback intent detection using simple keyword matching when the model fails.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing intent and confidence score
        """
        if not text.strip():
            return [{"intent": "general_knowledge", "score": 0.5}]
            
        text_lower = text.lower()
        matched_intents = []
        
        # Check for greeting patterns
        greeting_words = ['hi', 'hello', 'hey', 'greetings', 'howdy']
        if any(word in text_lower.split() for word in greeting_words):
            matched_intents.append({"intent": "greeting", "score": 0.9})
            
        # Check for goodbye patterns
        goodbye_words = ['bye', 'goodbye', 'see you', 'take care']
        if any(word in text_lower for word in goodbye_words):
            matched_intents.append({"intent": "goodbye", "score": 0.9})
            
        # Check for help patterns
        help_phrases = ['help', 'what can you do', 'how does this work']
        if any(phrase in text_lower for phrase in help_phrases):
            matched_intents.append({"intent": "help", "score": 0.9})
            
        # Check for definition patterns
        definition_phrases = ['what is', 'define', 'meaning of', 'explain', 'what does']
        if any(phrase in text_lower for phrase in definition_phrases):
            matched_intents.append({"intent": "definition", "score": 0.85})
            
        # Check for science questions
        science_terms = ['newton', 'law of motion', 'physics', 'chemistry', 'biology', 
                        'quantum', 'relativity', 'gravity', 'evolution', 'theory of']
        if any(term in text_lower for term in science_terms):
            matched_intents.append({"intent": "science_question", "score": 0.85})
            
        # Check for general knowledge questions
        question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
        if any(text_lower.startswith(word) for word in question_words):
            matched_intents.append({"intent": "general_knowledge", "score": 0.8})
            
        # If no specific intent matched, return a default general response
        if not matched_intents:
            matched_intents.append({"intent": "general_knowledge", "score": 0.7})
            
        return matched_intents
    
    def get_primary_intent(self, text: str, threshold: float = 0.3) -> Optional[Dict[str, Union[str, float]]]:
        """
        Get the primary (most confident) intent from the text.
        
        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold (0-1)
            
        Returns:
            Dictionary with intent details or None if no intent meets the threshold
        """
        intents = self.detect_intents(text, threshold)
        if intents:
            # If the top intent has very low confidence, check for specific keywords
            if intents[0]['score'] < 0.3:
                keyword_intent = self._fallback_intent_detection(text)
                if keyword_intent:
                    return keyword_intent[0]
            return intents[0]
        return None
    
    def add_custom_intent(self, 
                         intent_name: str, 
                         examples: List[str], 
                         description: str = "") -> bool:
        """
        Add a custom intent to the processor.
        
        Args:
            intent_name: Name of the new intent
            examples: List of example phrases for this intent
            description: Optional description of the intent
            
        Returns:
            True if intent was added successfully, False otherwise
        """
        if intent_name in self.intents:
            logger.warning(f"Intent '{intent_name}' already exists")
            return False
            
        self.intents[intent_name] = {
            "examples": examples,
            "description": description
        }
        
        # Reinitialize the model with the new intents
        try:
            self._initialize_model()
            return True
        except Exception as e:
            logger.error(f"Failed to update model with new intent: {str(e)}")
            del self.intents[intent_name]
            return False

# Singleton instance for easy import
nlp_processor = NLPProcessor()

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = NLPProcessor()
    
    # Example queries
    test_queries = [
        "Hello, how are you?",
        "Tell me a joke",
        "What's the weather like today?",
        "Goodbye!",
        "Can you help me with something?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        intents = processor.detect_intents(query)
        for intent in intents:
            print(f"- {intent['intent']} ({intent['score']:.2f}): {intent['description']}")
