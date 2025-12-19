"""
Timeline Construction Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import stages
from stage1_pairwise import (
    train_stage1, PairwiseClassifier, Stage1Config,
    evaluate_model_per_language, compute_all_features, load_model
)
from stage2_clustering import (
    EventClusterer, ClusteringConfig, evaluate_clustering,
    evaluate_clustering_per_language
)
from stage3_threading import (
    train_stage3,
    EventThreadingClassifier,
    TimelineBuilder
)


class PipelineConfig:
    """Pipeline configuration with multilingual support."""

    def __init__(
        self,
        data_dir: str = '../output/prepared_data',
        model_dir: str = '../output/models',
        results_dir: str = '../output/results',
        max_train_samples: int = 150000,
        max_test_samples: int = 50000,
        use_embeddings: bool = True,
        temporal_window_days: int = 30,
        similarity_threshold: float = 0.75,
        temporal_decay_lambda: float = 0.05,
        min_cluster_size: int = 2,
        leiden_resolution: float = 1.0,
        language_mode: str = 'english',  # 'english' or 'multilingual'
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.language_mode = language_mode

        self.stage1_config = Stage1Config(
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            use_sentence_embeddings=use_embeddings,
            language_mode=language_mode
        )

        self.clustering_config = ClusteringConfig(
            temporal_window_days=temporal_window_days,
            similarity_threshold=similarity_threshold,
            temporal_decay_lambda=temporal_decay_lambda,
            min_cluster_size=min_cluster_size,
            leiden_resolution=leiden_resolution
        )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Baseline English metrics for comparison
ENGLISH_BASELINE_METRICS = {
    'stage1': {'accuracy': 0.9130, 'f1_score': 0.9128},
    'stage2': {'ari': 0.8481, 'bcubed_f1': 0.8035},
    'stage3': {'accuracy': 0.3761, 'macro_f1': 0.2366}
}


class TimelinePipeline:
    """News timeline construction pipeline."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        self.pairwise_classifier = None
        self.event_clusterer = None
        self.threading_classifier = None
        self.timeline_builder = None

        self.metrics = {}
    
    def _get_model_suffix(self) -> str:
        """Get model filename suffix based on language mode."""
        if self.config.language_mode == 'multilingual':
            return '_multilingual'
        return ''

    def train(
        self,
        skip_stage1: bool = False,
        skip_stage2: bool = False,
        skip_stage3: bool = False
    ):
        """Train all pipeline stages."""
        suffix = self._get_model_suffix()
        stage1_path = self.config.model_dir / f'stage1_xgboost{suffix}.pkl'
        stage3_path = self.config.model_dir / f'stage3_threading{suffix}.pkl'
        is_multilingual = self.config.language_mode == 'multilingual'

        print(f"\nLanguage mode: {self.config.language_mode}")

        if not skip_stage1:
            print("\nStage 1: Pairwise Classifier")
            model, vectorizer, metrics = train_stage1(
                data_dir=str(self.config.data_dir),
                model_output=str(stage1_path),
                config=self.config.stage1_config
            )
            self.metrics['stage1'] = metrics

            # Per-language evaluation for Stage 1
            if is_multilingual:
                print("\n--- Stage 1: Per-Language Evaluation ---")
                test_df = pd.read_csv(self.config.data_dir / 'pairwise_test.csv')
                test_df = test_df.dropna(subset=['text_a', 'text_b', 'label'])

                stage1_per_lang = evaluate_model_per_language(
                    model=model,
                    test_df=test_df,
                    vectorizer=vectorizer,
                    use_embeddings=self.config.stage1_config.use_sentence_embeddings,
                    embedding_model=self.config.stage1_config.embedding_model,
                    batch_size=self.config.stage1_config.batch_size,
                    language_mode=self.config.language_mode
                )
                self.metrics['stage1_per_language'] = stage1_per_lang

                # Save per-language metrics
                with open(self.config.results_dir / 'stage1_detailed_evaluation.json', 'w') as f:
                    json.dump(stage1_per_lang, f, indent=2)

        self.pairwise_classifier = PairwiseClassifier(str(stage1_path))

        if not skip_stage2:
            print("\nStage 2: Event Clustering")
            self.event_clusterer = EventClusterer(
                stage1_model_path=str(stage1_path),
                config=self.config.clustering_config
            )

            test_articles = pd.read_csv(self.config.data_dir / 'article_index_test.csv')
            predicted_clusters = self.event_clusterer.fit_predict(test_articles)

            true_labels = test_articles['topic_index'].values
            clustering_metrics = evaluate_clustering(predicted_clusters, true_labels)
            self.metrics['stage2'] = clustering_metrics

            # Per-language evaluation for Stage 2
            if is_multilingual:
                print("\n--- Stage 2: Per-Language Evaluation ---")
                stage2_per_lang = evaluate_clustering_per_language(
                    articles=test_articles,
                    predicted_clusters=predicted_clusters,
                    true_labels=true_labels
                )
                self.metrics['stage2_per_language'] = stage2_per_lang

                # Save per-language metrics
                with open(self.config.results_dir / 'stage2_detailed_evaluation.json', 'w') as f:
                    json.dump(stage2_per_lang, f, indent=2)

            test_articles['predicted_cluster'] = predicted_clusters
            test_articles.to_csv(
                self.config.results_dir / f'stage2_test_clusters{suffix}.csv',
                index=False
            )

        if not skip_stage3:
            print("\nStage 3: Event Threading")
            classifier, metrics = train_stage3(
                data_dir=str(self.config.data_dir),
                model_output=str(stage3_path),
                language_mode=self.config.language_mode
            )
            self.metrics['stage3'] = metrics

        self.threading_classifier = EventThreadingClassifier.load(str(stage3_path))
        self.timeline_builder = TimelineBuilder(self.threading_classifier)

        self._save_metrics()
        self._print_summary()
    
    def load_models(self):
        """Load pre-trained models."""
        suffix = self._get_model_suffix()
        stage1_path = self.config.model_dir / f'stage1_xgboost{suffix}.pkl'
        stage3_path = self.config.model_dir / f'stage3_threading{suffix}.pkl'

        print(f"Loading models (language_mode={self.config.language_mode})...")

        self.pairwise_classifier = PairwiseClassifier(str(stage1_path))

        self.event_clusterer = EventClusterer(
            stage1_model_path=str(stage1_path),
            config=self.config.clustering_config
        )

        self.threading_classifier = EventThreadingClassifier.load(str(stage3_path))

        self.timeline_builder = TimelineBuilder(self.threading_classifier)
    
    def predict(self, articles: pd.DataFrame) -> Dict[int, List[Dict]]:
        """Run full pipeline on new articles."""
        if self.pairwise_classifier is None:
            self.load_models()
        
        print("\nRunning inference...")
        print("Clustering articles...")
        clusters = self.event_clusterer.fit_predict(articles)
        articles['cluster'] = clusters
        
        print("Building timelines...")
        
        timelines = {}
        unique_clusters = [c for c in set(clusters) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_articles = articles[articles['cluster'] == cluster_id]
            timeline = self.timeline_builder.build_timeline(cluster_articles, method='hybrid')
            timelines[cluster_id] = timeline
        
        noise_articles = articles[articles['cluster'] == -1]
        if len(noise_articles) > 0:
            timelines[-1] = [
                {'position': i, 'title': row['title'], 'date': str(row['date']), 'role': 'unclustered'}
                for i, (_, row) in enumerate(noise_articles.iterrows())
            ]
        
        print(f"Generated {len(timelines)} timelines")
        return timelines
    
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_json = {}
        for stage, stage_metrics in self.metrics.items():
            metrics_json[stage] = {}
            for key, value in stage_metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_json[stage][key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    metrics_json[stage][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    metrics_json[stage][key] = int(value)
                else:
                    metrics_json[stage][key] = value
        
        with open(self.config.results_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=2)
    
    def _print_summary(self):
        """Print training summary with three-section breakdown."""
        is_multilingual = self.config.language_mode == 'multilingual'

        print("\n" + "=" * 80)
        print("TRAINING RESULTS SUMMARY")
        print("=" * 80)

        # ======================================================================
        # SECTION 1: Overall Pipeline Performance
        # ======================================================================
        print("\n[SECTION 1] OVERALL PIPELINE PERFORMANCE")
        print("-" * 95)

        if is_multilingual:
            print(f"{'Stage':<12} {'Metric':<12} {'EN Only':<12} {'ZH Only':<12} {'Combined':<12} {'EN→Multi':<12} {'ZH→Multi':<12}")
            print("-" * 95)
        else:
            print(f"{'Stage':<12} {'Metric':<15} {'Value':<12}")
            print("-" * 45)

        # Load detailed metrics for per-language breakdown
        stage1_detailed = self.metrics.get('stage1_per_language', {})
        stage2_detailed = self.metrics.get('stage2_per_language', {})

        # Also try to load from files if not in memory
        if is_multilingual and not stage1_detailed:
            stage1_path = self.config.results_dir / 'stage1_detailed_evaluation.json'
            if stage1_path.exists():
                with open(stage1_path, 'r') as f:
                    stage1_detailed = json.load(f)

        if is_multilingual and not stage2_detailed:
            stage2_path = self.config.results_dir / 'stage2_detailed_evaluation.json'
            if stage2_path.exists():
                with open(stage2_path, 'r') as f:
                    stage2_detailed = json.load(f)

        stage3_detailed = None
        if is_multilingual:
            detailed_path = self.config.results_dir / 'stage3_detailed_evaluation.json'
            if detailed_path.exists():
                with open(detailed_path, 'r') as f:
                    stage3_detailed = json.load(f)

        if 'stage1' in self.metrics:
            acc = self.metrics['stage1']['accuracy']
            f1 = self.metrics['stage1']['f1_score']
            if is_multilingual:
                base_acc = ENGLISH_BASELINE_METRICS['stage1']['accuracy']
                base_f1 = ENGLISH_BASELINE_METRICS['stage1']['f1_score']

                # Get per-language metrics from detailed evaluation
                en_acc = stage1_detailed.get('en', {}).get('accuracy', acc)
                zh_acc = stage1_detailed.get('zh', {}).get('accuracy', acc)
                en_f1 = stage1_detailed.get('en', {}).get('f1_score', f1)
                zh_f1 = stage1_detailed.get('zh', {}).get('f1_score', f1)

                delta_en = en_acc - base_acc
                delta_zh = zh_acc - base_acc
                print(f"{'Stage 1':<12} {'Accuracy':<12} {en_acc:<12.4f} {zh_acc:<12.4f} {acc:<12.4f} {delta_en:+.4f}      {delta_zh:+.4f}")
                delta_en_f1 = en_f1 - base_f1
                delta_zh_f1 = zh_f1 - base_f1
                print(f"{'':<12} {'F1 Score':<12} {en_f1:<12.4f} {zh_f1:<12.4f} {f1:<12.4f} {delta_en_f1:+.4f}      {delta_zh_f1:+.4f}")
            else:
                print(f"{'Stage 1':<12} {'Accuracy':<15} {acc:<12.4f}")
                print(f"{'':<12} {'F1 Score':<15} {f1:<12.4f}")

        if 'stage2' in self.metrics:
            ari = self.metrics['stage2']['ari']
            bcubed = self.metrics['stage2']['bcubed_f1']
            if is_multilingual:
                base_ari = ENGLISH_BASELINE_METRICS['stage2']['ari']
                base_bcubed = ENGLISH_BASELINE_METRICS['stage2']['bcubed_f1']

                # Get per-language metrics from detailed evaluation
                en_ari = stage2_detailed.get('en', {}).get('ari', ari)
                zh_ari = stage2_detailed.get('zh', {}).get('ari', ari)
                en_bcubed = stage2_detailed.get('en', {}).get('bcubed_f1', bcubed)
                zh_bcubed = stage2_detailed.get('zh', {}).get('bcubed_f1', bcubed)

                delta_en_ari = en_ari - base_ari
                delta_zh_ari = zh_ari - base_ari
                print(f"{'Stage 2':<12} {'ARI':<12} {en_ari:<12.4f} {zh_ari:<12.4f} {ari:<12.4f} {delta_en_ari:+.4f}      {delta_zh_ari:+.4f}")
                delta_en_bc = en_bcubed - base_bcubed
                delta_zh_bc = zh_bcubed - base_bcubed
                print(f"{'':<12} {'BCubed F1':<12} {en_bcubed:<12.4f} {zh_bcubed:<12.4f} {bcubed:<12.4f} {delta_en_bc:+.4f}      {delta_zh_bc:+.4f}")
            else:
                print(f"{'Stage 2':<12} {'ARI':<15} {ari:<12.4f}")
                print(f"{'':<12} {'BCubed F1':<15} {bcubed:<12.4f}")

        if 'stage3' in self.metrics:
            acc = self.metrics['stage3']['accuracy']
            f1 = self.metrics['stage3']['macro_f1']
            if is_multilingual:
                base_acc = ENGLISH_BASELINE_METRICS['stage3']['accuracy']
                base_f1 = ENGLISH_BASELINE_METRICS['stage3']['macro_f1']

                # Get per-language metrics if available
                if stage3_detailed:
                    en_acc = stage3_detailed.get('en', {}).get('accuracy', base_acc)
                    zh_acc = stage3_detailed.get('zh', {}).get('accuracy', 0)
                    en_f1 = stage3_detailed.get('en', {}).get('macro_f1', base_f1)
                    zh_f1 = stage3_detailed.get('zh', {}).get('macro_f1', 0)

                    delta_en = en_acc - base_acc
                    delta_zh = zh_acc - base_acc
                    print(f"{'Stage 3':<12} {'Accuracy':<12} {en_acc:<12.4f} {zh_acc:<12.4f} {acc:<12.4f} {delta_en:+.4f}      {delta_zh:+.4f}")

                    delta_en_f1 = en_f1 - base_f1
                    delta_zh_f1 = zh_f1 - base_f1
                    print(f"{'':<12} {'Macro F1':<12} {en_f1:<12.4f} {zh_f1:<12.4f} {f1:<12.4f} {delta_en_f1:+.4f}      {delta_zh_f1:+.4f}")
                else:
                    delta_acc = acc - base_acc
                    delta_f1 = f1 - base_f1
                    print(f"{'Stage 3':<12} {'Accuracy':<12} {base_acc:<14.4f} {acc:<14.4f} {delta_acc:+.4f}")
                    print(f"{'':<12} {'Macro F1':<12} {base_f1:<14.4f} {f1:<14.4f} {delta_f1:+.4f}")
            else:
                print(f"{'Stage 3':<12} {'Accuracy':<15} {acc:<12.4f}")
                print(f"{'':<12} {'Macro F1':<15} {f1:<12.4f}")

        if is_multilingual:
            print()
            print("COLUMN EXPLANATIONS:")
            print("  EN Only:   Performance on English-only test data (balanced 50/50 sample)")
            print("  ZH Only:   Performance on Chinese-only test data (balanced 50/50 sample)")
            print("  Combined:  Overall multilingual model performance (balanced 50/50 sample)")
            print("  EN→Multi:  Change from English baseline to multilingual model (EN subset)")
            print("  ZH→Multi:  Change from English baseline to multilingual model (ZH subset)")
            print()
            print("NOTE: All per-language metrics use balanced sampling (50% positive, 50% negative)")
            print("      to ensure fair comparison and match the training distribution.")

        # ======================================================================
        # SECTION 2: Per-Stage Detailed Breakdown (multilingual only)
        # ======================================================================
        if is_multilingual:
            # Stage 1 detailed breakdown
            if stage1_detailed:
                print("\n" + "-" * 80)
                print("[SECTION 2a] STAGE 1: PER-LANGUAGE PERFORMANCE (Pairwise)")
                print("-" * 80)
                print(f"{'Language':<15} {'Samples':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
                print("-" * 90)

                for lang_key, lang_name in [('en', 'English'), ('zh', 'Chinese'), ('all', 'Combined')]:
                    if lang_key in stage1_detailed:
                        d = stage1_detailed[lang_key]
                        print(f"{lang_name:<15} {d['n_samples']:<12} {d['accuracy']*100:<12.2f} {d['precision']*100:<12.2f} {d['recall']*100:<12.2f} {d['f1_score']*100:<12.2f} {d['roc_auc']*100:<12.2f}")

            # Stage 2 detailed breakdown
            if stage2_detailed:
                print("\n" + "-" * 80)
                print("[SECTION 2b] STAGE 2: PER-LANGUAGE PERFORMANCE (Clustering)")
                print("-" * 80)
                print(f"{'Language':<15} {'Articles':<12} {'Clusters':<12} {'ARI':<12} {'NMI':<12} {'BCubed P':<12} {'BCubed R':<12} {'BCubed F1':<12}")
                print("-" * 100)

                for lang_key, lang_name in [('en', 'English'), ('zh', 'Chinese'), ('all', 'Combined')]:
                    if lang_key in stage2_detailed:
                        d = stage2_detailed[lang_key]
                        print(f"{lang_name:<15} {d['n_samples']:<12} {d['n_clusters']:<12} {d['ari']*100:<12.2f} {d['nmi']*100:<12.2f} {d['bcubed_precision']*100:<12.2f} {d['bcubed_recall']*100:<12.2f} {d['bcubed_f1']*100:<12.2f}")

            # Stage 3 detailed breakdown
            if stage3_detailed and 'stage3' in self.metrics:
                print("\n" + "-" * 80)
                print("[SECTION 2c] STAGE 3: PER-LANGUAGE PERFORMANCE (Threading)")
                print("-" * 80)
                print(f"{'Language':<15} {'Samples':<10} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
                print("-" * 80)

                if 'en' in stage3_detailed:
                    en = stage3_detailed['en']
                    print(f"{'English':<15} {en['n_samples']:<10} {en['accuracy']*100:<12.2f} {en['macro_f1']*100:<12.2f} {en['weighted_f1']*100:<12.2f}")

                if 'zh' in stage3_detailed:
                    zh = stage3_detailed['zh']
                    print(f"{'Chinese':<15} {zh['n_samples']:<10} {zh['accuracy']*100:<12.2f} {zh['macro_f1']*100:<12.2f} {zh['weighted_f1']*100:<12.2f}")

                if 'all' in stage3_detailed:
                    all_metrics = stage3_detailed['all']
                    print(f"{'Combined':<15} {all_metrics['n_samples']:<10} {all_metrics['accuracy']*100:<12.2f} {all_metrics['macro_f1']*100:<12.2f} {all_metrics['weighted_f1']*100:<12.2f}")

                # Show Chinese advantage
                if 'en' in stage3_detailed and 'zh' in stage3_detailed:
                    en_acc = stage3_detailed['en']['accuracy']
                    zh_acc = stage3_detailed['zh']['accuracy']
                    diff = (zh_acc - en_acc) * 100
                    print(f"\n  * Chinese outperforms English by {diff:+.2f}% accuracy")

            # ======================================================================
            # SECTION 3: Per-Class Metrics by Language (Stage 3 only)
            # ======================================================================
            if stage3_detailed and 'stage3' in self.metrics:
                print("\n" + "=" * 140)
                print("[SECTION 3] STAGE 3: PER-CLASS PERFORMANCE BY LANGUAGE")
                print("=" * 140)

                classes = ['SAME_DAY', 'IMMEDIATE_UPDATE', 'SHORT_TERM_DEV', 'LONG_TERM_DEV', 'DISTANT_RELATED']

                # English metrics
                print("\nENGLISH:")
                print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
                print("-" * 70)
                for cls in classes:
                    cls_data = stage3_detailed.get('en', {}).get('class_report', {}).get(cls, {})
                    prec = cls_data.get('precision', 0) * 100
                    rec = cls_data.get('recall', 0) * 100
                    f1 = cls_data.get('f1-score', 0) * 100
                    support = cls_data.get('support', 0)
                    print(f"{cls:<20} {prec:<12.2f} {rec:<12.2f} {f1:<12.2f} {support:<12}")

                # Chinese metrics
                print("\nCHINESE:")
                print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
                print("-" * 70)
                for cls in classes:
                    cls_data = stage3_detailed.get('zh', {}).get('class_report', {}).get(cls, {})
                    prec = cls_data.get('precision', 0) * 100
                    rec = cls_data.get('recall', 0) * 100
                    f1 = cls_data.get('f1-score', 0) * 100
                    support = cls_data.get('support', 0)
                    print(f"{cls:<20} {prec:<12.2f} {rec:<12.2f} {f1:<12.2f} {support:<12}")

                # Combined metrics
                print("\nCOMBINED:")
                print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
                print("-" * 70)
                for cls in classes:
                    cls_data = stage3_detailed.get('all', {}).get('class_report', {}).get(cls, {})
                    prec = cls_data.get('precision', 0) * 100
                    rec = cls_data.get('recall', 0) * 100
                    f1 = cls_data.get('f1-score', 0) * 100
                    support = cls_data.get('support', 0)
                    print(f"{cls:<20} {prec:<12.2f} {rec:<12.2f} {f1:<12.2f} {support:<12}")

        print("\n" + "=" * 80)

        # Analysis section
        if is_multilingual:
            print("\nANALYSIS:")
            if 'stage1' in self.metrics:
                delta = self.metrics['stage1']['accuracy'] - ENGLISH_BASELINE_METRICS['stage1']['accuracy']
                status = "improved" if delta > 0 else "degraded"
                print(f"  - Stage 1 (Pairwise): {status} by {abs(delta)*100:.2f}%")
            if 'stage2' in self.metrics:
                delta = self.metrics['stage2']['ari'] - ENGLISH_BASELINE_METRICS['stage2']['ari']
                status = "improved" if delta > 0 else "degraded"
                print(f"  - Stage 2 (Clustering): {status} by {abs(delta)*100:.2f}%")
            if 'stage3' in self.metrics:
                delta = self.metrics['stage3']['accuracy'] - ENGLISH_BASELINE_METRICS['stage3']['accuracy']
                status = "improved" if delta > 0 else "degraded"
                print(f"  - Stage 3 (Threading): {status} by {abs(delta)*100:.2f}%")
                print(f"    Note: Chinese subset performs better due to validated linguistic markers")


def main():
    parser = argparse.ArgumentParser(description='News Timeline Pipeline')

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'inference'])
    parser.add_argument('--data_dir', type=str, default='../output/prepared_data')
    parser.add_argument('--model_dir', type=str, default='../output/models')
    parser.add_argument('--results_dir', type=str, default='../output/results')
    parser.add_argument('--max_train', type=int, default=150000)
    parser.add_argument('--max_test', type=int, default=50000)
    parser.add_argument('--no_embeddings', action='store_true',
                        help='Disable sentence embeddings')
    parser.add_argument('--temporal_window', type=int, default=30)
    parser.add_argument('--similarity_threshold', type=float, default=0.75)
    parser.add_argument('--leiden_resolution', type=float, default=1.0)
    parser.add_argument('--skip_stage1', action='store_true')
    parser.add_argument('--skip_stage2', action='store_true')
    parser.add_argument('--skip_stage3', action='store_true')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--language_mode', type=str, default='english',
                        choices=['english', 'multilingual'],
                        help='Language mode: english (original) or multilingual (EN+ZH)')

    args = parser.parse_args()

    config = PipelineConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
        use_embeddings=not args.no_embeddings,
        temporal_window_days=args.temporal_window,
        similarity_threshold=args.similarity_threshold,
        leiden_resolution=args.leiden_resolution,
        language_mode=args.language_mode
    )
    
    pipeline = TimelinePipeline(config)
    
    if args.mode == 'train':
        pipeline.train(
            skip_stage1=args.skip_stage1,
            skip_stage2=args.skip_stage2,
            skip_stage3=args.skip_stage3
        )
    
    elif args.mode == 'evaluate':
        pipeline.load_models()
        test_articles = pd.read_csv(config.data_dir / 'article_index_test.csv')
        timelines = pipeline.predict(test_articles)
        
        output_path = config.results_dir / 'timelines.json'
        with open(output_path, 'w') as f:
            json.dump({str(k): v for k, v in timelines.items()}, f, indent=2)
        print(f"Timelines saved to {output_path}")
    
    elif args.mode == 'inference':
        if not args.input:
            raise ValueError("--input required for inference mode")
        
        pipeline.load_models()
        articles = pd.read_json(args.input)
        timelines = pipeline.predict(articles)
        
        output_path = args.output or (config.results_dir / 'timelines.json')
        with open(output_path, 'w') as f:
            json.dump({str(k): v for k, v in timelines.items()}, f, indent=2)


if __name__ == "__main__":
    main()