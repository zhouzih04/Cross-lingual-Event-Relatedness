"""
Linguistic Marker Analysis for Stage 3 Event Threading

This script performs systematic analysis of linguistic markers in news headlines
to identify statistically significant indicators of temporal relationships.

Methodology:
1. Extract candidate markers from training data
2. Compute frequency per temporal relation class
3. Apply chi-squared test for statistical significance
4. Filter markers by p-value threshold and effect size
5. Generate report with all statistics for paper

Author: Generated for CIS5300 Milestone 3
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple, Set
import json
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import jieba for Chinese segmentation
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available, Chinese analysis will be limited")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Statistical thresholds
P_VALUE_THRESHOLD = 0.05  # Significance level
MIN_OCCURRENCES = 10  # Minimum times a marker must appear
MIN_EFFECT_SIZE = 0.05  # Minimum Cramér's V for practical significance (lowered to capture more patterns)

# Temporal relation classes
TEMPORAL_CLASSES = [
    'SAME_DAY',
    'IMMEDIATE_UPDATE',
    'SHORT_TERM_DEV',
    'LONG_TERM_DEV',
    'DISTANT_RELATED'
]

# =============================================================================
# CANDIDATE MARKERS (Theory-driven, from journalism/NLP literature)
# =============================================================================

# English candidate markers based on temporal discourse literature
ENGLISH_CANDIDATE_MARKERS = {
    'update_indicators': [
        'update', 'updated', 'latest', 'new', 'now', 'breaking',
        'continues', 'continuing', 'ongoing', 'still', 'again',
        'another', 'more', 'further', 'additional', 'fresh',
        'developing', 'unfolding', 'emerging'
    ],
    'backward_references': [
        'after', 'following', 'since', 'aftermath', 'wake',
        'response', 'responds', 'responded', 'responding',
        'reaction', 'reacts', 'reacted',
        'result', 'results', 'resulting',
        'consequence', 'consequences',
        'fallout', 'repercussions'
    ],
    'breaking_news': [
        'breaking', 'just in', 'alert', 'urgent', 'developing',
        'exclusive', 'live', 'happening now', 'flash'
    ],
    'time_references': [
        'today', 'yesterday', 'tomorrow', 'tonight', 'morning',
        'week', 'weeks', 'month', 'months', 'year', 'years',
        'day', 'days', 'hour', 'hours', 'later', 'earlier',
        'ago', 'ahead'
    ],
    'statement_verbs': [
        'says', 'said', 'tells', 'told', 'announces', 'announced',
        'confirms', 'confirmed', 'claims', 'claimed',
        'reports', 'reported', 'reveals', 'revealed',
        'declares', 'declared', 'states', 'stated'
    ],
    'completion_markers': [
        'completed', 'finished', 'concluded', 'ended', 'resolved',
        'approved', 'confirmed', 'finalized', 'done', 'over'
    ],
    'process_verbs': [
        'investigating', 'investigation', 'probe', 'probing',
        'examining', 'reviewing', 'analyzing', 'assessing',
        'considering', 'deliberating'
    ],
    'negation_denial': [
        'denies', 'denied', 'refuses', 'refused', 'rejects', 'rejected',
        'dismisses', 'dismissed', 'not', "n't", 'no'
    ]
}

# Chinese candidate markers
CHINESE_CANDIDATE_MARKERS = {
    'update_indicators': [
        '续', '继续', '又', '再次', '再', '最新', '最新消息',
        '更新', '刚刚', '仍', '还在', '持续', '连续'
    ],
    'backward_references': [
        '回应', '反应', '后', '之后', '随后', '结果', '已被',
        '因', '因为', '由于', '导致', '引发', '造成'
    ],
    'breaking_news': [
        '突发', '刚刚', '最新消息', '紧急', '速报', '快讯',
        '突然', '爆', '独家'
    ],
    'time_references': [
        '今天', '昨天', '明天', '今晚', '上午', '下午',
        '周', '月', '年', '日', '天', '小时', '分钟',
        '之前', '之后', '以来', '以后'
    ],
    'statement_verbs': [
        '称', '表示', '宣布', '声明', '说', '指出',
        '透露', '报道', '确认', '证实', '发布'
    ],
    'completion_markers': [
        '已被', '完成', '确认', '批准', '通过', '结束',
        '落幕', '告终', '收官'
    ],
    'process_verbs': [
        '调查', '处理', '处罚', '问责', '审查', '检查',
        '核实', '追查', '审理'
    ],
    'negation_denial': [
        '否认', '拒绝', '驳斥', '反驳', '不', '未', '没有'
    ]
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def detect_language(text: str) -> str:
    """Detect if text is Chinese or English based on character ratio."""
    if not text:
        return 'en'
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(' ', ''))
    if total_chars == 0:
        return 'en'
    return 'zh' if chinese_chars / total_chars > 0.3 else 'en'


def count_marker_occurrences(
    df: pd.DataFrame,
    markers: Dict[str, List[str]],
    text_column: str = 'text_b',
    language: str = 'en'
) -> pd.DataFrame:
    """
    Count occurrences of each marker category in the dataset.

    Returns DataFrame with columns: marker_category, marker, temporal_relation, count
    """
    results = []

    for category, marker_list in markers.items():
        for marker in marker_list:
            # Count per temporal class
            for temp_class in TEMPORAL_CLASSES:
                class_df = df[df['temporal_relation'] == temp_class]

                if language == 'zh':
                    # For Chinese, direct substring match
                    count = class_df[text_column].str.contains(
                        marker, case=False, na=False, regex=False
                    ).sum()
                else:
                    # For English, word boundary match
                    pattern = r'\b' + re.escape(marker) + r'\b'
                    count = class_df[text_column].str.contains(
                        pattern, case=False, na=False, regex=True
                    ).sum()

                results.append({
                    'category': category,
                    'marker': marker,
                    'temporal_relation': temp_class,
                    'count': count,
                    'class_total': len(class_df),
                    'frequency': count / len(class_df) if len(class_df) > 0 else 0
                })

    return pd.DataFrame(results)


def compute_chi_squared(
    marker_counts: pd.DataFrame,
    marker: str
) -> Dict:
    """
    Compute chi-squared test for a single marker.

    Tests if the marker distribution across temporal classes is significantly
    different from expected (uniform distribution).

    Returns dict with chi2 statistic, p-value, Cramér's V, and per-class frequencies.
    """
    marker_df = marker_counts[marker_counts['marker'] == marker]

    if len(marker_df) == 0:
        return None

    # Create contingency table: [has_marker, no_marker] x [temporal_classes]
    observed = []
    for temp_class in TEMPORAL_CLASSES:
        row = marker_df[marker_df['temporal_relation'] == temp_class]
        if len(row) > 0:
            has_marker = row['count'].values[0]
            total = row['class_total'].values[0]
            no_marker = total - has_marker
            observed.append([has_marker, no_marker])
        else:
            observed.append([0, 0])

    observed = np.array(observed)

    # Skip if marker never appears
    total_occurrences = observed[:, 0].sum()
    if total_occurrences < MIN_OCCURRENCES:
        return None

    # Chi-squared test
    try:
        chi2, p_value, dof, expected = chi2_contingency(observed)
    except ValueError:
        return None

    # Cramér's V (effect size)
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0

    # Per-class frequencies
    class_frequencies = {}
    for i, temp_class in enumerate(TEMPORAL_CLASSES):
        if observed[i, :].sum() > 0:
            class_frequencies[temp_class] = observed[i, 0] / observed[i, :].sum()
        else:
            class_frequencies[temp_class] = 0

    # Find which class has highest frequency
    peak_class = max(class_frequencies, key=class_frequencies.get)

    return {
        'marker': marker,
        'total_occurrences': int(total_occurrences),
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'significant': p_value < P_VALUE_THRESHOLD and cramers_v >= MIN_EFFECT_SIZE,
        'class_frequencies': class_frequencies,
        'peak_class': peak_class,
        'peak_frequency': class_frequencies[peak_class]
    }


def analyze_markers(
    df: pd.DataFrame,
    markers: Dict[str, List[str]],
    language: str = 'en'
) -> pd.DataFrame:
    """
    Full analysis pipeline for a set of markers.

    Returns DataFrame with statistical results for each marker.
    """
    print(f"\nAnalyzing {language.upper()} markers...")
    print(f"Dataset size: {len(df)} pairs")

    # Count occurrences
    counts_df = count_marker_occurrences(df, markers, language=language)

    # Compute statistics for each marker
    results = []
    all_markers = set(counts_df['marker'].unique())

    for marker in all_markers:
        stats = compute_chi_squared(counts_df, marker)
        if stats:
            # Find category
            category = counts_df[counts_df['marker'] == marker]['category'].iloc[0]
            stats['category'] = category
            results.append(stats)

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Sort by significance and effect size
        results_df = results_df.sort_values(
            ['significant', 'cramers_v', 'p_value'],
            ascending=[False, False, True]
        )

    return results_df


def generate_validated_marker_lists(
    en_results: pd.DataFrame,
    zh_results: pd.DataFrame
) -> Dict:
    """
    Generate validated marker lists based on statistical analysis.

    Only includes markers that:
    1. Pass chi-squared test (p < 0.05)
    2. Have sufficient effect size (Cramér's V >= 0.1)
    3. Appear at least MIN_OCCURRENCES times
    """
    validated = {
        'english': defaultdict(list),
        'chinese': defaultdict(list)
    }

    # English
    if len(en_results) > 0:
        sig_en = en_results[en_results['significant'] == True]
        for _, row in sig_en.iterrows():
            validated['english'][row['category']].append({
                'marker': row['marker'],
                'peak_class': row['peak_class'],
                'peak_frequency': row['peak_frequency'],
                'p_value': row['p_value'],
                'cramers_v': row['cramers_v']
            })

    # Chinese
    if len(zh_results) > 0:
        sig_zh = zh_results[zh_results['significant'] == True]
        for _, row in sig_zh.iterrows():
            validated['chinese'][row['category']].append({
                'marker': row['marker'],
                'peak_class': row['peak_class'],
                'peak_frequency': row['peak_frequency'],
                'p_value': row['p_value'],
                'cramers_v': row['cramers_v']
            })

    return validated


def print_analysis_report(
    results_df: pd.DataFrame,
    language: str
):
    """Print formatted analysis report."""
    print(f"\n{'='*70}")
    print(f"LINGUISTIC MARKER ANALYSIS: {language.upper()}")
    print(f"{'='*70}")

    if len(results_df) == 0:
        print("No markers analyzed.")
        return

    sig_count = results_df['significant'].sum()
    total_count = len(results_df)

    print(f"\nTotal markers analyzed: {total_count}")
    print(f"Statistically significant: {sig_count} ({100*sig_count/total_count:.1f}%)")
    print(f"Significance threshold: p < {P_VALUE_THRESHOLD}, Cramér's V >= {MIN_EFFECT_SIZE}")

    print(f"\n{'-'*70}")
    print("SIGNIFICANT MARKERS (sorted by effect size):")
    print(f"{'-'*70}")

    sig_df = results_df[results_df['significant'] == True]

    if len(sig_df) == 0:
        print("No significant markers found.")
        return

    for _, row in sig_df.head(20).iterrows():
        print(f"\n  {row['marker']}")
        print(f"    Category: {row['category']}")
        print(f"    Occurrences: {row['total_occurrences']}")
        print(f"    Chi-squared: {row['chi2']:.2f}, p-value: {row['p_value']:.4f}")
        print(f"    Effect size (Cramér's V): {row['cramers_v']:.3f}")
        print(f"    Peak class: {row['peak_class']} ({100*row['peak_frequency']:.1f}%)")

        # Show all class frequencies
        freq_str = ", ".join([
            f"{k}: {100*v:.1f}%"
            for k, v in sorted(row['class_frequencies'].items(), key=lambda x: -x[1])
        ])
        print(f"    All classes: {freq_str}")

    print(f"\n{'-'*70}")
    print("NON-SIGNIFICANT MARKERS (top 10 by occurrence):")
    print(f"{'-'*70}")

    nonsig_df = results_df[results_df['significant'] == False].head(10)
    for _, row in nonsig_df.iterrows():
        print(f"  {row['marker']}: n={row['total_occurrences']}, p={row['p_value']:.4f}, V={row['cramers_v']:.3f}")


def save_results(
    en_results: pd.DataFrame,
    zh_results: pd.DataFrame,
    validated_markers: Dict,
    output_dir: str = '../output/results'
):
    """Save all analysis results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    if len(en_results) > 0:
        en_results.to_csv(output_path / 'marker_analysis_english.csv', index=False)

    if len(zh_results) > 0:
        zh_results.to_csv(output_path / 'marker_analysis_chinese.csv', index=False)

    # Save validated markers as JSON
    # Convert to serializable format
    validated_serializable = {}
    for lang, categories in validated_markers.items():
        validated_serializable[lang] = {}
        for cat, markers in categories.items():
            validated_serializable[lang][cat] = markers

    with open(output_path / 'validated_markers.json', 'w', encoding='utf-8') as f:
        json.dump(validated_serializable, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}/")


def generate_feature_code(validated_markers: Dict) -> str:
    """Generate Python code for the validated features."""

    code_lines = [
        "# =============================================================================",
        "# VALIDATED LINGUISTIC MARKERS",
        "# Generated by analyze_linguistic_markers.py",
        "# All markers passed chi-squared test (p < 0.05) with Cramér's V >= 0.1",
        "# =============================================================================",
        ""
    ]

    # English markers
    code_lines.append("# English markers (statistically validated)")
    for category, markers in validated_markers['english'].items():
        if markers:
            marker_list = [m['marker'] for m in markers]
            code_lines.append(f"{category.upper()}_EN = {marker_list}")

    code_lines.append("")

    # Chinese markers
    code_lines.append("# Chinese markers (statistically validated)")
    for category, markers in validated_markers['chinese'].items():
        if markers:
            marker_list = [m['marker'] for m in markers]
            code_lines.append(f"{category.upper()}_ZH = {marker_list}")

    return "\n".join(code_lines)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Run full linguistic marker analysis."""

    print("="*70)
    print("LINGUISTIC MARKER ANALYSIS FOR TEMPORAL RELATION CLASSIFICATION")
    print("="*70)
    print(f"\nStatistical thresholds:")
    print(f"  - P-value threshold: {P_VALUE_THRESHOLD}")
    print(f"  - Minimum occurrences: {MIN_OCCURRENCES}")
    print(f"  - Minimum effect size (Cramér's V): {MIN_EFFECT_SIZE}")

    # Load training data
    data_path = Path('../output/prepared_data')

    print("\nLoading training data...")
    train_df = pd.read_csv(data_path / 'threading_train.csv')
    print(f"Loaded {len(train_df)} training pairs")

    # Add language column if not present
    if 'language' not in train_df.columns:
        train_df['language'] = train_df['text_a'].apply(detect_language)

    # Split by language
    en_df = train_df[train_df['language'] == 'en'].copy()
    zh_df = train_df[train_df['language'] == 'zh'].copy()

    print(f"English pairs: {len(en_df)}")
    print(f"Chinese pairs: {len(zh_df)}")

    # Class distribution
    print("\nClass distribution (training):")
    for temp_class in TEMPORAL_CLASSES:
        en_count = len(en_df[en_df['temporal_relation'] == temp_class])
        zh_count = len(zh_df[zh_df['temporal_relation'] == temp_class])
        print(f"  {temp_class}: EN={en_count}, ZH={zh_count}")

    # Analyze English markers
    en_results = pd.DataFrame()
    if len(en_df) > 0:
        en_results = analyze_markers(en_df, ENGLISH_CANDIDATE_MARKERS, language='en')
        print_analysis_report(en_results, 'English')

    # Analyze Chinese markers
    zh_results = pd.DataFrame()
    if len(zh_df) > 0 and JIEBA_AVAILABLE:
        zh_results = analyze_markers(zh_df, CHINESE_CANDIDATE_MARKERS, language='zh')
        print_analysis_report(zh_results, 'Chinese')

    # Generate validated marker lists
    validated_markers = generate_validated_marker_lists(en_results, zh_results)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: VALIDATED MARKERS FOR FEATURE ENGINEERING")
    print("="*70)

    print("\nENGLISH:")
    for category, markers in validated_markers['english'].items():
        if markers:
            marker_strs = [f"{m['marker']} ({m['peak_class']}, V={m['cramers_v']:.2f})"
                          for m in markers[:5]]
            print(f"  {category}: {', '.join(marker_strs)}")

    print("\nCHINESE:")
    for category, markers in validated_markers['chinese'].items():
        if markers:
            marker_strs = [f"{m['marker']} ({m['peak_class']}, V={m['cramers_v']:.2f})"
                          for m in markers[:5]]
            print(f"  {category}: {', '.join(marker_strs)}")

    # Save results
    save_results(en_results, zh_results, validated_markers)

    # Generate code snippet
    print("\n" + "="*70)
    print("GENERATED CODE FOR STAGE 3 FEATURES:")
    print("="*70)
    print(generate_feature_code(validated_markers))

    return en_results, zh_results, validated_markers


if __name__ == "__main__":
    en_results, zh_results, validated_markers = main()
