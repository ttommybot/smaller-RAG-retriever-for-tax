# -*- coding: utf-8 -*-
"""
Analyze triplet data quality with score distribution and text length stats.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import statistics

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_triplets(file_path: str) -> List[Dict[str, Any]]:
    """Load triplet data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_scores(triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze score distribution."""
    if not triplets:
        return {}

    positive_scores = [t['positive_score'] for t in triplets]
    negative_scores = [t['negative_score'] for t in triplets]
    score_gaps = [t['positive_score'] - t['negative_score'] for t in triplets]

    return {
        'positive_scores': {
            'min': min(positive_scores),
            'max': max(positive_scores),
            'mean': statistics.mean(positive_scores),
            'median': statistics.median(positive_scores),
            'stdev': statistics.stdev(positive_scores) if len(positive_scores) > 1 else 0
        },
        'negative_scores': {
            'min': min(negative_scores),
            'max': max(negative_scores),
            'mean': statistics.mean(negative_scores),
            'median': statistics.median(negative_scores),
            'stdev': statistics.stdev(negative_scores) if len(negative_scores) > 1 else 0
        },
        'score_gaps': {
            'min': min(score_gaps),
            'max': max(score_gaps),
            'mean': statistics.mean(score_gaps),
            'median': statistics.median(score_gaps),
            'stdev': statistics.stdev(score_gaps) if len(score_gaps) > 1 else 0
        }
    }


def analyze_text_length(triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze text length statistics."""
    if not triplets:
        return {}

    query_lengths = [len(t['query']) for t in triplets]
    positive_lengths = [len(t['positive']) for t in triplets]
    negative_lengths = [len(t['negative']) for t in triplets]

    return {
        'query': {
            'min': min(query_lengths),
            'max': max(query_lengths),
            'mean': statistics.mean(query_lengths),
            'median': statistics.median(query_lengths)
        },
        'positive': {
            'min': min(positive_lengths),
            'max': max(positive_lengths),
            'mean': statistics.mean(positive_lengths),
            'median': statistics.median(positive_lengths)
        },
        'negative': {
            'min': min(negative_lengths),
            'max': max(negative_lengths),
            'mean': statistics.mean(negative_lengths),
            'median': statistics.median(negative_lengths)
        }
    }


def analyze_quality(triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate data quality by score gap."""
    if not triplets:
        return {}

    high_quality = 0  # gap > 0.3
    medium_quality = 0  # 0.2 < gap <= 0.3
    low_quality = 0  # gap <= 0.2

    for t in triplets:
        gap = t['positive_score'] - t['negative_score']
        if gap > 0.3:
            high_quality += 1
        elif gap > 0.2:
            medium_quality += 1
        else:
            low_quality += 1

    total = len(triplets)

    return {
        'total': total,
        'high_quality': {
            'count': high_quality,
            'percentage': round(high_quality / total * 100, 2)
        },
        'medium_quality': {
            'count': medium_quality,
            'percentage': round(medium_quality / total * 100, 2)
        },
        'low_quality': {
            'count': low_quality,
            'percentage': round(low_quality / total * 100, 2)
        }
    }


def check_data_quality(data_dir: str = "data/training") -> Dict[str, Any]:
    """Run complete data quality check."""
    data_path = Path(data_dir)

    print("=" * 60)
    print("Data Quality Check")
    print("=" * 60)

    print("\nStep 1: Loading data...")
    triplets_file = data_path / "triplets.json"
    triplets_with_scores_file = data_path / "triplets_with_scores.json"
    stats_file = data_path / "generation_stats.json"

    if not triplets_file.exists():
        print(f"File not found: {triplets_file}")
        return {}

    triplets = load_triplets(str(triplets_file))
    print(f"Loaded {len(triplets)} triplets")

    if triplets_with_scores_file.exists():
        triplets_with_scores = load_triplets(str(triplets_with_scores_file))
        print(f"Loaded {len(triplets_with_scores)} triplets with scores")
    else:
        triplets_with_scores = []
        print("Triplets with scores not found")

    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            generation_stats = json.load(f)
        print(f"Loaded generation stats")
    else:
        generation_stats = {}
        print("Generation stats not found")

    report = {
        'basic_stats': {
            'total_triplets': len(triplets),
            'generation_stats': generation_stats
        }
    }

    if triplets_with_scores:
        print("\nStep 2: Analyzing score distribution...")
        score_analysis = analyze_scores(triplets_with_scores)
        report['score_analysis'] = score_analysis

        print(f"  Positive score: {score_analysis['positive_scores']['mean']:.4f} ± {score_analysis['positive_scores']['stdev']:.4f}")
        print(f"  Negative score: {score_analysis['negative_scores']['mean']:.4f} ± {score_analysis['negative_scores']['stdev']:.4f}")
        print(f"  Score gap: {score_analysis['score_gaps']['mean']:.4f} ± {score_analysis['score_gaps']['stdev']:.4f}")

    print("\nStep 3: Analyzing text length...")
    length_analysis = analyze_text_length(triplets)
    report['length_analysis'] = length_analysis

    print(f"  Query avg length: {length_analysis['query']['mean']:.1f} chars")
    print(f"  Positive avg length: {length_analysis['positive']['mean']:.1f} chars")
    print(f"  Negative avg length: {length_analysis['negative']['mean']:.1f} chars")

    if triplets_with_scores:
        print("\nStep 4: Evaluating data quality...")
        quality_analysis = analyze_quality(triplets_with_scores)
        report['quality_analysis'] = quality_analysis

        print(f"  High quality (gap>0.3): {quality_analysis['high_quality']['count']} ({quality_analysis['high_quality']['percentage']}%)")
        print(f"  Medium quality (0.2<gap≤0.3): {quality_analysis['medium_quality']['count']} ({quality_analysis['medium_quality']['percentage']}%)")
        print(f"  Low quality (gap≤0.2): {quality_analysis['low_quality']['count']} ({quality_analysis['low_quality']['percentage']}%)")

    print("\nStep 5: Saving quality report...")
    report_file = data_path / "quality_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Quality report saved: {report_file}")

    readable_report_file = data_path / "quality_report.md"
    with open(readable_report_file, 'w', encoding='utf-8') as f:
        f.write("# Training Data Quality Report\n\n")
        f.write(f"**Generated**: {Path(triplets_file).stat().st_mtime}\n\n")
        f.write("---\n\n")

        f.write("## 1. Basic Statistics\n\n")
        f.write(f"- **Total Triplets**: {len(triplets)}\n")
        if generation_stats:
            f.write(f"- **Processed Queries**: {generation_stats.get('processed_queries', 0)}\n")
            f.write(f"- **Avg per Query**: {generation_stats.get('total_triplets', 0) / max(generation_stats.get('processed_queries', 1), 1):.2f} triplets\n")
        f.write("\n")

        if triplets_with_scores:
            f.write("## 2. Score Distribution\n\n")
            f.write("### Positive Scores\n")
            f.write(f"- Mean: {score_analysis['positive_scores']['mean']:.4f}\n")
            f.write(f"- Median: {score_analysis['positive_scores']['median']:.4f}\n")
            f.write(f"- Range: [{score_analysis['positive_scores']['min']:.4f}, {score_analysis['positive_scores']['max']:.4f}]\n\n")

            f.write("### Negative Scores\n")
            f.write(f"- Mean: {score_analysis['negative_scores']['mean']:.4f}\n")
            f.write(f"- Median: {score_analysis['negative_scores']['median']:.4f}\n")
            f.write(f"- Range: [{score_analysis['negative_scores']['min']:.4f}, {score_analysis['negative_scores']['max']:.4f}]\n\n")

            f.write("### Score Gap\n")
            f.write(f"- Mean gap: {score_analysis['score_gaps']['mean']:.4f}\n")
            f.write(f"- Median gap: {score_analysis['score_gaps']['median']:.4f}\n")
            f.write(f"- Range: [{score_analysis['score_gaps']['min']:.4f}, {score_analysis['score_gaps']['max']:.4f}]\n\n")

        f.write("## 3. Text Length\n\n")
        f.write(f"- **Query**: avg {length_analysis['query']['mean']:.1f} chars\n")
        f.write(f"- **Positive**: avg {length_analysis['positive']['mean']:.1f} chars\n")
        f.write(f"- **Negative**: avg {length_analysis['negative']['mean']:.1f} chars\n\n")

        if triplets_with_scores:
            f.write("## 4. Quality Assessment\n\n")
            f.write(f"- **High quality** (gap>0.3): {quality_analysis['high_quality']['count']} ({quality_analysis['high_quality']['percentage']}%)\n")
            f.write(f"- **Medium quality** (0.2<gap≤0.3): {quality_analysis['medium_quality']['count']} ({quality_analysis['medium_quality']['percentage']}%)\n")
            f.write(f"- **Low quality** (gap≤0.2): {quality_analysis['low_quality']['count']} ({quality_analysis['low_quality']['percentage']}%)\n\n")

        f.write("## 5. Recommendations\n\n")
        if triplets_with_scores:
            if quality_analysis['high_quality']['percentage'] >= 60:
                f.write("Data quality is good, ready for training.\n")
            elif quality_analysis['high_quality']['percentage'] >= 40:
                f.write("Data quality is moderate, consider adjusting thresholds.\n")
            else:
                f.write("Data quality is low, regenerate data or adjust parameters.\n")

    print(f"Readable report saved: {readable_report_file}")

    print("\n" + "=" * 60)
    print("Quality check complete")
    print("=" * 60)

    return report
