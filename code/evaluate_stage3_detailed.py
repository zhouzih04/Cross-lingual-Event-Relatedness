"""
Detailed Stage 3 Evaluation Script

This script provides per-language evaluation and detailed analysis
of the temporal threading classification results.

Outputs:
- Per-language performance metrics
- Per-class performance breakdown
- Feature importance analysis
- Confusion matrix analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_recall_fscore_support
)
from collections import defaultdict
import re

# Import from stage3
from stage3_threading import (
    extract_threading_features_fixed,
    TEMPORAL_RELATIONS
)


def load_model(model_path: str):
    """Load trained Stage 3 model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def detect_language(text: str) -> str:
    """Detect language of text."""
    if not text or pd.isna(text):
        return 'en'
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', str(text)))
    total_chars = len(str(text).replace(' ', ''))
    if total_chars == 0:
        return 'en'
    return 'zh' if chinese_chars / total_chars > 0.3 else 'en'


def evaluate_per_language(test_df, model_data, language_mode='multilingual'):
    """Evaluate model performance separately for each language."""

    model = model_data['model']
    vectorizer = model_data.get('vectorizer')
    label_encoder = model_data['label_encoder']

    # Add language column if not present
    if 'language' not in test_df.columns:
        test_df['language'] = test_df['text_a'].apply(detect_language)

    results = {}

    for lang in ['en', 'zh', 'all']:
        if lang == 'all':
            df = test_df
        else:
            df = test_df[test_df['language'] == lang]

        if len(df) == 0:
            continue

        # Extract features
        features_list = []
        for _, row in df.iterrows():
            features = extract_threading_features_fixed(
                row['text_a'], row['text_b'],
                tfidf_vectorizer=vectorizer,
                language_mode=language_mode
            )
            features_list.append(features)

        X = pd.DataFrame(features_list)
        y_true = label_encoder.transform(df['temporal_relation'])

        # Predict
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        results[lang] = {
            'n_samples': len(df),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'class_report': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': list(label_encoder.classes_)
        }

    return results


def print_detailed_report(results):
    """Print detailed evaluation report."""

    print("="*70)
    print("DETAILED STAGE 3 EVALUATION REPORT")
    print("="*70)

    print("\n" + "-"*70)
    print("OVERALL PERFORMANCE BY LANGUAGE")
    print("-"*70)

    print(f"\n{'Language':<12} {'Samples':<10} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-"*58)

    for lang, data in results.items():
        lang_name = {'en': 'English', 'zh': 'Chinese', 'all': 'Combined'}[lang]
        print(f"{lang_name:<12} {data['n_samples']:<10} {data['accuracy']:<12.4f} {data['macro_f1']:<12.4f} {data['weighted_f1']:<12.4f}")

    # Per-class breakdown
    for lang in ['en', 'zh']:
        if lang not in results:
            continue

        lang_name = 'English' if lang == 'en' else 'Chinese'
        data = results[lang]

        print(f"\n" + "-"*70)
        print(f"PER-CLASS PERFORMANCE: {lang_name.upper()}")
        print("-"*70)

        print(f"\n{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-"*66)

        for class_name in data['class_names']:
            if class_name in data['class_report']:
                cls = data['class_report'][class_name]
                print(f"{class_name:<20} {cls['precision']:<12.3f} {cls['recall']:<12.3f} {cls['f1-score']:<12.3f} {cls['support']:<10}")

    # Analysis
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)

    if 'en' in results and 'zh' in results:
        en_acc = results['en']['accuracy']
        zh_acc = results['zh']['accuracy']

        if zh_acc > en_acc:
            print(f"\n✓ Chinese ({zh_acc:.2%}) outperforms English ({en_acc:.2%}) by {(zh_acc-en_acc)*100:.1f}%")
            print("  This suggests Chinese-specific features are providing additional signal.")
        else:
            print(f"\n✓ English ({en_acc:.2%}) outperforms Chinese ({zh_acc:.2%}) by {(en_acc-zh_acc)*100:.1f}%")

    # Confusion matrix analysis
    print("\n" + "-"*70)
    print("CONFUSION MATRIX ANALYSIS")
    print("-"*70)

    if 'all' in results:
        cm = np.array(results['all']['confusion_matrix'])
        class_names = results['all']['class_names']

        # Calculate per-class predictions
        pred_counts = cm.sum(axis=0)
        true_counts = cm.sum(axis=1)

        print(f"\n{'Class':<20} {'True Count':<12} {'Pred Count':<12} {'Over/Under':<15}")
        print("-"*59)

        for i, cls in enumerate(class_names):
            diff = pred_counts[i] - true_counts[i]
            indicator = "OVER" if diff > 0 else "UNDER" if diff < 0 else "BALANCED"
            print(f"{cls:<20} {true_counts[i]:<12} {pred_counts[i]:<12} {indicator} ({diff:+d})")


def main():
    """Run detailed evaluation."""

    # Paths
    data_path = Path('../output/prepared_data')
    model_path = Path('../output/models/stage3_threading_multilingual.pkl')

    print("Loading test data...")
    test_df = pd.read_csv(data_path / 'threading_test.csv')
    print(f"Loaded {len(test_df)} test pairs")

    print("Loading model...")
    model_data = load_model(str(model_path))

    print("Evaluating...")
    results = evaluate_per_language(test_df, model_data, language_mode='multilingual')

    print_detailed_report(results)

    # Save results
    output_path = Path('../output/results')
    output_path.mkdir(exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for lang, data in results.items():
        results_json[lang] = {
            'n_samples': data['n_samples'],
            'accuracy': data['accuracy'],
            'macro_f1': data['macro_f1'],
            'weighted_f1': data['weighted_f1'],
            'class_report': data['class_report']
        }

    with open(output_path / 'stage3_detailed_evaluation.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to {output_path / 'stage3_detailed_evaluation.json'}")

    return results


if __name__ == "__main__":
    results = main()
