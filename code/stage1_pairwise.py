"""
Stage 1: Pairwise Relatedness Classifier
Supports both English-only and multilingual (English + Chinese) modes.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import pickle
from typing import Tuple, List, Optional, Set
from pathlib import Path
import warnings

from language_utils import (
    detect_language,
    tokenize_text,
    tokenize_text_for_tfidf,
    preprocess_text_multilingual,
    extract_words,
    get_bigrams_multilingual,
    get_first_n_tokens,
    extract_capitalized_words_multilingual,
    extract_numbers,
    get_embedding_model_name,
    EMBEDDING_MODELS
)


class Stage1Config:
    """Stage 1 configuration."""

    def __init__(
        self,
        max_train_samples: int = 150000,
        max_val_samples: int = 15000,
        max_test_samples: int = 50000,
        tfidf_max_features: int = 5000,
        batch_size: int = 5000,
        use_sentence_embeddings: bool = True,
        embedding_model: str = None,  # Auto-selected based on language_mode
        language_mode: str = 'english',  # 'english' or 'multilingual'
        random_seed: int = 42
    ):
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.tfidf_max_features = tfidf_max_features
        self.batch_size = batch_size
        self.use_sentence_embeddings = use_sentence_embeddings
        self.language_mode = language_mode
        # Auto-select embedding model based on language mode if not specified
        if embedding_model is None:
            self.embedding_model = get_embedding_model_name(language_mode)
        else:
            self.embedding_model = embedding_model
        self.random_seed = random_seed


def preprocess_text(text: str) -> str:
    """Normalize text."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def extract_numbers(text: str) -> Set[str]:
    """Extract all numbers from text."""
    return set(re.findall(r'\b\d+\b', str(text)))


def extract_capitalized_words(text: str) -> Set[str]:
    """Extract capitalized words."""
    if pd.isna(text):
        return set()
    words = re.findall(r'\b[A-Z][a-z]+\b', str(text))
    return set(w.lower() for w in words)


def get_bigrams(text: str) -> Set[Tuple[str, str]]:
    """Extract word bigrams."""
    words = text.split()
    if len(words) < 2:
        return set()
    return set(zip(words[:-1], words[1:]))


def get_first_n_words(text: str, n: int = 3) -> List[str]:
    """Get first N words."""
    words = text.split()
    return words[:n] if len(words) >= n else words


class EmbeddingCache:
    """Sentence embedding cache."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.cache = {}
    
    def _load_model(self):
        """Load embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
    
    def encode(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts to embeddings with caching."""
        self._load_model()
        
        # Check which texts need encoding
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            if text not in self.cache:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode new texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode, 
                batch_size=batch_size,
                show_progress_bar=len(texts_to_encode) > 1000
            )
            for text, emb in zip(texts_to_encode, new_embeddings):
                self.cache[text] = emb
        
        # Gather all embeddings
        embeddings = np.array([self.cache[text] for text in texts])
        return embeddings


_embedding_cache = None

def get_embedding_cache(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingCache:
    """Get embedding cache."""
    global _embedding_cache
    if _embedding_cache is None or _embedding_cache.model_name != model_name:
        _embedding_cache = EmbeddingCache(model_name)
    return _embedding_cache


def compute_embedding_similarity(
    texts_a: List[str],
    texts_b: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> np.ndarray:
    """Compute embedding similarity features."""
    cache = get_embedding_cache(model_name)
    
    all_texts = list(set(texts_a) | set(texts_b))
    cache.encode(all_texts)
    
    features = []
    for text_a, text_b in zip(texts_a, texts_b):
        emb_a = cache.cache[text_a]
        emb_b = cache.cache[text_b]
        
        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        cosine = dot / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
        
        euclidean = np.linalg.norm(emb_a - emb_b) / np.sqrt(len(emb_a))
        manhattan = np.sum(np.abs(emb_a - emb_b)) / len(emb_a)
        
        features.append([cosine, euclidean, manhattan, dot])
    
    return np.array(features, dtype=np.float32)


def compute_handcrafted_features(
    texts_a: pd.Series,
    texts_b: pd.Series,
    vectorizer: TfidfVectorizer,
    batch_size: int = 5000,
    language_mode: str = 'english'
) -> np.ndarray:
    """Compute hand-crafted features with language-aware processing."""
    n_samples = len(texts_a)
    all_features = []

    original_texts_a = texts_a.copy()
    original_texts_b = texts_b.copy()

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)

        batch_orig_a = original_texts_a.iloc[start_idx:end_idx]
        batch_orig_b = original_texts_b.iloc[start_idx:end_idx]
        batch_texts_a = texts_a.iloc[start_idx:end_idx].apply(preprocess_text)
        batch_texts_b = texts_b.iloc[start_idx:end_idx].apply(preprocess_text)

        # TF-IDF for this batch
        batch_tfidf_a = vectorizer.transform(batch_texts_a).toarray()
        batch_tfidf_b = vectorizer.transform(batch_texts_b).toarray()

        batch_features = []

        for i in range(len(batch_texts_a)):
            text_a = batch_texts_a.iloc[i]
            text_b = batch_texts_b.iloc[i]
            orig_a = str(batch_orig_a.iloc[i]) if not pd.isna(batch_orig_a.iloc[i]) else ""
            orig_b = str(batch_orig_b.iloc[i]) if not pd.isna(batch_orig_b.iloc[i]) else ""

            # Language-aware word extraction
            if language_mode == 'multilingual':
                words_a = extract_words(text_a)
                words_b = extract_words(text_b)
            else:
                words_a = set(text_a.split())
                words_b = set(text_b.split())

            intersection = len(words_a & words_b)
            union = len(words_a | words_b)

            jaccard = intersection / union if union > 0 else 0
            common_words = intersection
            word_count_a = len(words_a)
            word_count_b = len(words_b)
            word_count_diff = abs(word_count_a - word_count_b)
            word_count_ratio = min(word_count_a, word_count_b) / max(word_count_a, word_count_b) if max(word_count_a, word_count_b) > 0 else 0

            char_len_a = len(text_a)
            char_len_b = len(text_b)
            char_len_diff = abs(char_len_a - char_len_b)
            char_len_ratio = min(char_len_a, char_len_b) / max(char_len_a, char_len_b) if max(char_len_a, char_len_b) > 0 else 0

            vec_a = batch_tfidf_a[i]
            vec_b = batch_tfidf_b[i]

            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            cosine_sim = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
            euclidean_dist = np.linalg.norm(vec_a - vec_b)

            # Language-aware bigrams
            if language_mode == 'multilingual':
                bigrams_a = get_bigrams_multilingual(text_a)
                bigrams_b = get_bigrams_multilingual(text_b)
            else:
                bigrams_a = get_bigrams(text_a)
                bigrams_b = get_bigrams(text_b)
            bigram_intersection = len(bigrams_a & bigrams_b)
            bigram_union = len(bigrams_a | bigrams_b)
            bigram_overlap = bigram_intersection / bigram_union if bigram_union > 0 else 0

            # Language-aware first token extraction
            if language_mode == 'multilingual':
                tokens_a = tokenize_text(text_a)
                tokens_b = tokenize_text(text_b)
                first_a = tokens_a[0] if tokens_a else ""
                first_b = tokens_b[0] if tokens_b else ""
                first3_a = set(get_first_n_tokens(text_a, 3))
                first3_b = set(get_first_n_tokens(text_b, 3))
            else:
                first_a = text_a.split()[0] if text_a.split() else ""
                first_b = text_b.split()[0] if text_b.split() else ""
                first3_a = set(get_first_n_words(text_a, 3))
                first3_b = set(get_first_n_words(text_b, 3))
            first_word_match = 1 if first_a == first_b and first_a != "" else 0
            first3_intersection = len(first3_a & first3_b)
            first_3_words_overlap = first3_intersection / 3 if first3_a or first3_b else 0

            numbers_a = extract_numbers(orig_a)
            numbers_b = extract_numbers(orig_b)
            number_intersection = len(numbers_a & numbers_b)
            number_union = len(numbers_a | numbers_b)
            number_overlap = number_intersection / number_union if number_union > 0 else 0

            # Language-aware entity extraction
            if language_mode == 'multilingual':
                entities_a = extract_capitalized_words_multilingual(orig_a)
                entities_b = extract_capitalized_words_multilingual(orig_b)
            else:
                entities_a = extract_capitalized_words(orig_a)
                entities_b = extract_capitalized_words(orig_b)
            entity_intersection = len(entities_a & entities_b)
            entity_union = len(entities_a | entities_b)
            entity_overlap = entity_intersection / entity_union if entity_union > 0 else 0

            unique_words_a = len(words_a - words_b)
            unique_words_b = len(words_b - words_a)

            both_have_numbers = 1 if numbers_a and numbers_b else 0
            has_quote_a = 1 if '"' in orig_a or "'" in orig_a else 0
            has_quote_b = 1 if '"' in orig_b or "'" in orig_b else 0
            both_have_quotes = 1 if has_quote_a and has_quote_b else 0

            max_len = max(char_len_a, char_len_b)
            length_similarity = 1 - (char_len_diff / max_len) if max_len > 0 else 1

            prefix_match_len = 0
            min_len = min(len(text_a), len(text_b))
            for j in range(min_len):
                if text_a[j] == text_b[j]:
                    prefix_match_len += 1
                else:
                    break
            prefix_match_len = prefix_match_len / max_len if max_len > 0 else 0

            containment_a_in_b = len(words_a & words_b) / len(words_a) if len(words_a) > 0 else 0
            containment_b_in_a = len(words_a & words_b) / len(words_b) if len(words_b) > 0 else 0

            batch_features.append([
                jaccard, common_words, word_count_a, word_count_b, word_count_diff, word_count_ratio,
                char_len_a, char_len_b, char_len_diff, char_len_ratio,
                cosine_sim, euclidean_dist,
                bigram_overlap, first_word_match, first_3_words_overlap,
                number_overlap, entity_overlap, unique_words_a, unique_words_b,
                both_have_numbers, both_have_quotes, length_similarity,
                prefix_match_len, containment_a_in_b, containment_b_in_a
            ])

        all_features.extend(batch_features)
        del batch_tfidf_a, batch_tfidf_b, batch_features

        if (end_idx) % 20000 == 0 or end_idx == n_samples:
            print(f"  Processed {end_idx}/{n_samples} pairs...")

    return np.array(all_features, dtype=np.float32)


HANDCRAFTED_FEATURE_NAMES = [
    'jaccard', 'common_words', 'word_count_a', 'word_count_b', 'word_count_diff', 'word_count_ratio',
    'char_len_a', 'char_len_b', 'char_len_diff', 'char_len_ratio',
    'tfidf_cosine', 'tfidf_euclidean',
    'bigram_overlap', 'first_word_match', 'first_3_words_overlap',
    'number_overlap', 'entity_overlap', 'unique_words_a', 'unique_words_b',
    'both_have_numbers', 'both_have_quotes', 'length_similarity',
    'prefix_match_len', 'containment_a_in_b', 'containment_b_in_a'
]

EMBEDDING_FEATURE_NAMES = ['emb_cosine', 'emb_euclidean', 'emb_manhattan', 'emb_dot']


def compute_all_features(
    texts_a: pd.Series,
    texts_b: pd.Series,
    vectorizer: TfidfVectorizer,
    use_embeddings: bool = True,
    embedding_model: str = 'all-MiniLM-L6-v2',
    batch_size: int = 5000,
    language_mode: str = 'english'
) -> Tuple[np.ndarray, List[str]]:
    """Compute all features with language-aware processing."""
    hc_features = compute_handcrafted_features(
        texts_a, texts_b, vectorizer, batch_size, language_mode
    )
    feature_names = HANDCRAFTED_FEATURE_NAMES.copy()
    
    if use_embeddings:
        texts_a_list = texts_a.apply(lambda x: str(x) if not pd.isna(x) else "").tolist()
        texts_b_list = texts_b.apply(lambda x: str(x) if not pd.isna(x) else "").tolist()
        
        emb_features = compute_embedding_similarity(texts_a_list, texts_b_list, embedding_model)
        
        all_features = np.hstack([hc_features, emb_features])
        feature_names = feature_names + EMBEDDING_FEATURE_NAMES
    else:
        all_features = hc_features
    
    return all_features, feature_names


def sample_balanced_data(
    df: pd.DataFrame,
    max_samples: int,
    random_seed: int = 42
) -> pd.DataFrame:
    """Sample data while maintaining class balance."""
    if len(df) <= max_samples:
        return df
    
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    
    samples_per_class = max_samples // 2
    
    pos_sampled = pos_df.sample(n=min(samples_per_class, len(pos_df)), random_state=random_seed)
    neg_sampled = neg_df.sample(n=min(samples_per_class, len(neg_df)), random_state=random_seed)
    
    result = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"  Sampled {len(result)} pairs (was {len(df)})")
    print(f"  Class balance: {(result['label']==1).mean():.1%} positive")
    
    return result


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> xgb.XGBClassifier:
    """Train XGBoost model."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=8,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=15 if X_val is not None else None
    )
    
    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['verbose'] = False
    
    model.fit(X_train, y_train, **fit_params)
    return model


def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Unrelated', 'Related']))

    return metrics


def evaluate_model_per_language(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    use_embeddings: bool = True,
    embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    batch_size: int = 5000,
    language_mode: str = 'multilingual',
    max_samples_per_lang: int = 50000
) -> dict:
    """
    Evaluate model separately for each language in the test set.

    Applies balanced sampling within each language to ensure fair comparison
    with the overall "Combined" metric which uses balanced sampling.

    Args:
        model: Trained XGBoost model
        test_df: Test dataframe with 'text_a', 'text_b', 'label', and 'language' columns
        vectorizer: Fitted TF-IDF vectorizer
        use_embeddings: Whether to use sentence embeddings
        embedding_model: Name of embedding model
        batch_size: Batch size for feature computation
        language_mode: Language mode for feature extraction
        max_samples_per_lang: Maximum samples per language (with balanced sampling)

    Returns:
        Dictionary with per-language metrics: {'en': {...}, 'zh': {...}, 'all': {...}}
    """
    results = {}

    # Get unique languages
    if 'language' not in test_df.columns:
        print("Warning: 'language' column not found in test data. Returning overall metrics only.")
        return results

    languages = test_df['language'].unique()
    print(f"\nPer-language evaluation for Stage 1:")
    print(f"Languages found: {list(languages)}")
    print(f"Note: Applying balanced sampling (50/50 class ratio) for fair comparison")

    for lang in languages:
        lang_df = test_df[test_df['language'] == lang].copy()
        n_samples_raw = len(lang_df)

        if n_samples_raw == 0:
            continue

        # Skip "mixed" language pairs (cross-language pairs are always negative)
        if lang == 'mixed':
            print(f"\n  [MIXED] Skipping {n_samples_raw} cross-language pairs (always negative)")
            continue

        # Check if only one class is present
        n_classes = len(lang_df['label'].unique())
        if n_classes < 2:
            print(f"\n  [{lang.upper()}] Warning: Only one class present in {n_samples_raw} pairs, skipping")
            continue

        # Apply balanced sampling to match training distribution
        lang_df_sampled = sample_balanced_data(lang_df, max_samples_per_lang, random_seed=42)
        n_samples = len(lang_df_sampled)

        print(f"\n  [{lang.upper()}] Evaluating on {n_samples} pairs (sampled from {n_samples_raw})...")

        # Show class distribution before and after sampling
        pos_before = (lang_df['label'] == 1).sum()
        pos_after = (lang_df_sampled['label'] == 1).sum()
        print(f"    Class distribution: {pos_before}/{n_samples_raw} ({pos_before/n_samples_raw*100:.1f}%) → {pos_after}/{n_samples} ({pos_after/n_samples*100:.1f}%) positive")

        # Compute features for this language subset
        X_lang, _ = compute_all_features(
            lang_df_sampled['text_a'], lang_df_sampled['text_b'],
            vectorizer,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            batch_size=batch_size,
            language_mode=language_mode
        )
        y_lang = lang_df_sampled['label'].values.astype(np.int32)

        # Get predictions
        y_pred = model.predict(X_lang)
        y_pred_proba = model.predict_proba(X_lang)[:, 1]

        # Compute metrics (with zero_division handling)
        metrics = {
            'n_samples': n_samples,
            'n_samples_raw': n_samples_raw,
            'accuracy': float(accuracy_score(y_lang, y_pred)),
            'precision': float(precision_score(y_lang, y_pred, zero_division=0)),
            'recall': float(recall_score(y_lang, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_lang, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_lang, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_lang, y_pred).tolist()
        }

        results[lang] = metrics

        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")

    # Also compute overall metrics on balanced sample
    n_all_raw = len(test_df)

    # Apply balanced sampling to full dataset (excluding 'mixed' language pairs)
    test_df_no_mixed = test_df[test_df['language'] != 'mixed'].copy() if 'mixed' in languages else test_df.copy()
    test_df_sampled = sample_balanced_data(test_df_no_mixed, max_samples_per_lang, random_seed=42)
    n_all = len(test_df_sampled)

    print(f"\n  [ALL] Evaluating on {n_all} pairs (sampled from {n_all_raw})...")

    # Show class distribution
    pos_before_all = (test_df_no_mixed['label'] == 1).sum()
    pos_after_all = (test_df_sampled['label'] == 1).sum()
    print(f"    Class distribution: {pos_before_all}/{len(test_df_no_mixed)} ({pos_before_all/len(test_df_no_mixed)*100:.1f}%) → {pos_after_all}/{n_all} ({pos_after_all/n_all*100:.1f}%) positive")

    X_all, _ = compute_all_features(
        test_df_sampled['text_a'], test_df_sampled['text_b'],
        vectorizer,
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
        batch_size=batch_size,
        language_mode=language_mode
    )
    y_all = test_df_sampled['label'].values.astype(np.int32)
    y_pred_all = model.predict(X_all)
    y_pred_proba_all = model.predict_proba(X_all)[:, 1]

    results['all'] = {
        'n_samples': n_all,
        'n_samples_raw': n_all_raw,
        'accuracy': float(accuracy_score(y_all, y_pred_all)),
        'precision': float(precision_score(y_all, y_pred_all)),
        'recall': float(recall_score(y_all, y_pred_all)),
        'f1_score': float(f1_score(y_all, y_pred_all)),
        'roc_auc': float(roc_auc_score(y_all, y_pred_proba_all)),
        'confusion_matrix': confusion_matrix(y_all, y_pred_all).tolist()
    }

    print(f"    Accuracy: {results['all']['accuracy']:.4f}")
    print(f"    F1-Score: {results['all']['f1_score']:.4f}")
    print(f"    ROC-AUC: {results['all']['roc_auc']:.4f}")

    return results


def save_model(model, vectorizer, filepath: str, config: dict = None, feature_names: List[str] = None):
    """Save model."""
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'version': 'final_with_embeddings_v1',
        'feature_names': feature_names,
        'config': config
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(filepath: str) -> Tuple[xgb.XGBClassifier, TfidfVectorizer, dict]:
    """Load model."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['vectorizer'], model_data


class PairwiseClassifier:
    """Pairwise classifier with multilingual support."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.use_embeddings = True
        self.embedding_model = 'all-MiniLM-L6-v2'
        self.language_mode = 'english'

        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        """Load model."""
        self.model, self.vectorizer, self.metadata = load_model(model_path)

        if self.metadata.get('config'):
            self.use_embeddings = self.metadata['config'].get('use_sentence_embeddings', True)
            self.embedding_model = self.metadata['config'].get('embedding_model', 'all-MiniLM-L6-v2')
            self.language_mode = self.metadata['config'].get('language_mode', 'english')

    def predict_pair(self, text_a: str, text_b: str) -> Tuple[int, float]:
        """Predict if two texts are related."""
        df = pd.DataFrame({'text_a': [text_a], 'text_b': [text_b]})
        features, _ = compute_all_features(
            df['text_a'], df['text_b'],
            self.vectorizer,
            use_embeddings=self.use_embeddings,
            embedding_model=self.embedding_model,
            batch_size=1,
            language_mode=self.language_mode
        )

        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0, 1]

        return int(pred), float(prob)

    def predict_pairs_batch(
        self,
        texts_a: List[str],
        texts_b: List[str],
        batch_size: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction for multiple pairs."""
        df = pd.DataFrame({'text_a': texts_a, 'text_b': texts_b})

        features, _ = compute_all_features(
            df['text_a'], df['text_b'],
            self.vectorizer,
            use_embeddings=self.use_embeddings,
            embedding_model=self.embedding_model,
            batch_size=batch_size,
            language_mode=self.language_mode
        )

        preds = self.model.predict(features)
        probs = self.model.predict_proba(features)[:, 1]

        return np.array(preds), np.array(probs)


def train_stage1(
    data_dir: str = '../output/prepared_data',
    model_output: str = '../output/models/stage1_xgboost.pkl',
    config: Stage1Config = None
) -> Tuple[xgb.XGBClassifier, TfidfVectorizer, dict]:
    """Train Stage 1 pairwise classifier with multilingual support."""
    config = config or Stage1Config()

    data_path = Path(data_dir)

    print(f"Language mode: {config.language_mode}")
    print(f"Embedding model: {config.embedding_model}")

    print("Loading data...")
    train_df = pd.read_csv(data_path / 'xgboost_train.csv')
    val_df = pd.read_csv(data_path / 'xgboost_val.csv')
    test_df = pd.read_csv(data_path / 'xgboost_test.csv')

    print(f"Train: {len(train_df):,} pairs")
    print(f"Val: {len(val_df):,} pairs")
    print(f"Test: {len(test_df):,} pairs")

    train_df = train_df.dropna(subset=['text_a', 'text_b', 'label'])
    val_df = val_df.dropna(subset=['text_a', 'text_b', 'label'])
    test_df = test_df.dropna(subset=['text_a', 'text_b', 'label'])

    print("Sampling data...")
    train_df = sample_balanced_data(train_df, config.max_train_samples, config.random_seed)
    val_df = sample_balanced_data(val_df, config.max_val_samples, config.random_seed)
    test_df = sample_balanced_data(test_df, config.max_test_samples, config.random_seed)

    print("Fitting TF-IDF vectorizer...")
    all_texts = pd.concat([
        train_df['text_a'].apply(preprocess_text),
        train_df['text_b'].apply(preprocess_text)
    ])

    # Language-aware TF-IDF configuration
    if config.language_mode == 'multilingual':
        # For multilingual: use custom tokenizer, no English stop words
        vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            tokenizer=tokenize_text_for_tfidf,
            token_pattern=None  # Required when using custom tokenizer
        )
    else:
        # Original English-only configuration
        vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
    vectorizer.fit(all_texts)
    del all_texts
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("Computing features...")
    X_train, feature_names = compute_all_features(
        train_df['text_a'], train_df['text_b'], vectorizer,
        use_embeddings=config.use_sentence_embeddings,
        embedding_model=config.embedding_model,
        batch_size=config.batch_size,
        language_mode=config.language_mode
    )
    y_train = train_df['label'].values.astype(np.int32)

    X_val, _ = compute_all_features(
        val_df['text_a'], val_df['text_b'], vectorizer,
        use_embeddings=config.use_sentence_embeddings,
        embedding_model=config.embedding_model,
        batch_size=config.batch_size,
        language_mode=config.language_mode
    )
    y_val = val_df['label'].values.astype(np.int32)

    X_test, _ = compute_all_features(
        test_df['text_a'], test_df['text_b'], vectorizer,
        use_embeddings=config.use_sentence_embeddings,
        embedding_model=config.embedding_model,
        batch_size=config.batch_size,
        language_mode=config.language_mode
    )
    y_test = test_df['label'].values.astype(np.int32)
    
    print(f"Feature matrices - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    print("Training model...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("Saving model...")
    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, vectorizer, model_output, config.__dict__, feature_names)
    
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    print("Top 5 features:")
    for rank, idx in enumerate(sorted_idx[:5], 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {name}: {importance[idx]:.4f}")
    
    return model, vectorizer, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Stage 1 Pairwise Classifier')
    parser.add_argument('--data_dir', type=str, default='../output/prepared_data')
    parser.add_argument('--model_output', type=str, default='../output/models/stage1_xgboost.pkl')
    parser.add_argument('--max_train', type=int, default=150000)
    parser.add_argument('--max_test', type=int, default=50000)
    parser.add_argument('--no_embeddings', action='store_true', help='Disable sentence embeddings')
    parser.add_argument('--language_mode', type=str, default='english',
                        choices=['english', 'multilingual'],
                        help='Language mode: english (original) or multilingual (EN+ZH)')

    args = parser.parse_args()

    config = Stage1Config(
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
        use_sentence_embeddings=not args.no_embeddings,
        language_mode=args.language_mode
    )

    model, vectorizer, metrics = train_stage1(
        data_dir=args.data_dir,
        model_output=args.model_output,
        config=config
    )