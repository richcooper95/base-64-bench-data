#!/usr/bin/env python3
"""
Analyze base64 encoding/decoding benchmark results.

This script performs comprehensive analysis of eval results including:
- Threshold sweep analysis (accuracy vs threshold)
- Similarity distribution analysis
- Performance by data type
- Model comparison across metrics
"""

import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
import argparse
from inspect_ai.log import read_eval_log


def get_model_performance_ordering(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Get models ordered by overall performance at threshold=1.0 (descending).

    Returns:
        List of (model_name, accuracy_at_1_0) tuples, sorted by accuracy descending
    """
    model_performance_at_1_0 = []

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        accuracy_at_1_0 = (model_data['similarity'] >= 1.0).mean()
        model_performance_at_1_0.append((model, accuracy_at_1_0))

    # Sort models by performance at threshold=1.0 (descending)
    model_performance_at_1_0.sort(key=lambda x: x[1], reverse=True)
    return model_performance_at_1_0


def clean_model_name(full_name: str) -> str:
    """Clean model names by removing provider prefix."""
    return full_name.split('/')[-1]  # Take everything after the last slash


def get_model_color_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a consistent color mapping for all models based on overall performance ordering.
    Uses high-contrast colors that are easy to distinguish.

    Returns:
        Dictionary mapping model names to colors
    """
    # Get all models ordered by overall performance
    model_performance_ordering = get_model_performance_ordering(df)
    models = [model for model, _ in model_performance_ordering]

    # Define a curated list of high-contrast, distinguishable colors (avoiding light grey/yellow)
    high_contrast_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#17becf',  # cyan
        '#bcbd22',  # olive (darker yellow-green)
        '#ff1493',  # deep pink
        '#00ced1',  # dark turquoise
        '#ff4500',  # orange red
        '#32cd32',  # lime green
        '#ba55d3',  # medium orchid
        '#dc143c',  # crimson
        '#4169e1',  # royal blue
        '#ff8c00',  # dark orange
        '#228b22',  # forest green
        '#8b008b',  # dark magenta
        '#b8860b',  # dark goldenrod
    ]

    n_models = len(models)

    if n_models <= len(high_contrast_colors):
        # Use curated colors
        colors = [high_contrast_colors[i] for i in range(n_models)]
    else:
        # For more models, cycle through curated colors with varying saturation
        colors = []
        for i in range(n_models):
            base_color = high_contrast_colors[i % len(high_contrast_colors)]
            colors.append(base_color)

    # Create mapping
    color_map = {model: colors[i] for i, model in enumerate(models)}
    return color_map


def load_eval_results(logs_dir: str = "base64bench-logs/results") -> Dict[str, List]:
    """
    Load all evaluation results from the logs directory using Inspect AI's log reader.

    Returns:
        Dictionary mapping model names to lists of sample objects
    """
    results = {}

    # Find all .eval files
    pattern = os.path.join(logs_dir, "**", "*.eval")
    eval_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(eval_files)} eval files")

    for eval_file in eval_files:
        try:
            # Load the eval file using Inspect AI's reader
            log = read_eval_log(eval_file)

            # Extract model name from the eval log content
            model_name = "unknown"
            if hasattr(log, 'eval') and hasattr(log.eval, 'model') and log.eval.model:
                model_name = log.eval.model
            else:
                # Fallback to extracting from path if model info not in log
                path_parts = Path(eval_file).parts
                if len(path_parts) >= 3:
                    # Format: logs/provider/model/file.eval
                    provider = path_parts[-3]
                    model = path_parts[-2]
                    model_name = f"{provider}/{model}"

            if log.samples:
                if model_name not in results:
                    results[model_name] = []
                results[model_name].extend(log.samples)
                print(f"  {model_name}: {len(log.samples)} samples")

        except Exception as e:
            print(f"Error loading {eval_file}: {e}")

    return results


def extract_similarity_scores(results: Dict[str, List]) -> pd.DataFrame:
    """
    Extract similarity scores and metadata into a pandas DataFrame.

    Returns:
        DataFrame with columns: model, task, data_type, similarity, passed_at_1_0, input_length
    """
    rows = []

    for model_name, samples in results.items():
        for sample in samples:
            # sample.scores is a dict mapping scorer names to Score objects
            if hasattr(sample, 'scores') and sample.scores:
                # Look through all scorers (usually just one)
                for scorer_name, score_obj in sample.scores.items():
                    # Get task and data type from sample metadata
                    task_type = sample.metadata.get('task', 'unknown') if sample.metadata else 'unknown'
                    data_type = sample.metadata.get('type', 'unknown') if sample.metadata else 'unknown'

                    if hasattr(score_obj, 'metadata') and score_obj.metadata:
                        metadata = score_obj.metadata
                        # Extract key metrics from successful samples
                        similarity = metadata.get('similarity')
                        input_length = metadata.get('input_length', 0)
                        distance = metadata.get('distance', 0)
                    else:
                        # Failed samples (no metadata) - treat as 0.0 similarity to match Inspect's calculation
                        similarity = 0.0
                        input_length = 0
                        distance = float('inf')  # Large distance for failed attempts

                    if similarity is not None:
                        rows.append({
                            'model': model_name,
                            'task': task_type,
                            'data_type': data_type,
                            'similarity': similarity,
                            'passed_at_1_0': similarity >= 1.0,
                            'passed_at_95': similarity >= 0.95,
                            'passed_at_90': similarity >= 0.90,
                            'input_length': input_length,
                            'distance': distance,
                            'sample_id': getattr(sample, 'id', 'unknown'),
                            'score_value': getattr(score_obj, 'value', None),
                            'scorer': scorer_name,
                            'failed_decode': not (hasattr(score_obj, 'metadata') and score_obj.metadata)
                        })

    return pd.DataFrame(rows)


def plot_threshold_sweep(df: pd.DataFrame, save_path: str = None, color_map: Dict[str, str] = None):
    """Plot accuracy vs threshold for each model with zoom-in subplot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    thresholds = np.arange(0.0, 1.01, 0.05)
    # Create zoom thresholds for high-performance range
    thresholds_zoom = np.concatenate([
        np.arange(0.95, 1.0, 0.005),  # 0.95 to 0.995 with high resolution
        [1.0]  # Exactly 1.0
    ])

    # Get consistent model ordering
    model_performance_at_1_0 = get_model_performance_ordering(df)

    # Create color map if not provided
    if color_map is None:
        color_map = get_model_color_map(df)

    # Plot 1: Full range (0.0 - 1.0)
    for model, _ in model_performance_at_1_0:
        model_data = df[df['model'] == model]
        accuracies = []

        for thresh in thresholds:
            accuracy = (model_data['similarity'] >= thresh).mean()
            accuracies.append(accuracy)

        clean_name = clean_model_name(model)
        color = color_map.get(model, None)
        ax1.plot(thresholds, accuracies, marker='o', markersize=4, label=clean_name, alpha=0.7, linewidth=2.5, color=color)

    ax1.set_xlabel('Similarity Threshold', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.set_title('Accuracy vs Similarity Threshold (Full Range)', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed range (0.8 - 1.0)
    for model, _ in model_performance_at_1_0:
        model_data = df[df['model'] == model]
        accuracies_zoom = []

        for thresh in thresholds_zoom:
            accuracy = (model_data['similarity'] >= thresh).mean()
            accuracies_zoom.append(accuracy)

        clean_name = clean_model_name(model)
        color = color_map.get(model, None)
        ax2.plot(thresholds_zoom, accuracies_zoom, marker='o', markersize=4, label=clean_name, alpha=0.7, linewidth=2.5, color=color)

    ax2.set_xlabel('Similarity Threshold', fontsize=16)
    ax2.set_ylabel('Accuracy', fontsize=16)
    ax2.set_title('Accuracy vs Similarity Threshold (Zoomed: 0.95-1.0)', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.95, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_threshold_sweep_by_task(df: pd.DataFrame, encode_save_path: str = None, decode_save_path: str = None, color_map: Dict[str, str] = None):
    """Plot separate threshold sweep graphs for encode and decode tasks."""

    # Create color map if not provided (based on overall data for consistency)
    if color_map is None:
        color_map = get_model_color_map(df)

    def create_threshold_sweep_subplot(task_data, task_name, save_path, overall_df):
        """Create a threshold sweep plot for a specific task."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        thresholds = np.arange(0.0, 1.01, 0.05)
        # Create zoom thresholds for high-performance range
        thresholds_zoom = np.concatenate([
            np.arange(0.95, 1.0, 0.005),  # 0.95 to 0.995 with high resolution
            [1.0]  # Exactly 1.0
        ])

        # Get task-specific model ordering (based on performance at this task)
        model_performance_at_1_0 = get_model_performance_ordering(task_data)

        # Plot 1: Full range (0.0 - 1.0)
        for model, _ in model_performance_at_1_0:
            model_data = task_data[task_data['model'] == model]
            accuracies = []

            for thresh in thresholds:
                accuracy = (model_data['similarity'] >= thresh).mean()
                accuracies.append(accuracy)

            clean_name = clean_model_name(model)
            color = color_map.get(model, None)
            ax1.plot(thresholds, accuracies, marker='o', markersize=4, label=clean_name, alpha=0.7, linewidth=2.5, color=color)

        ax1.set_xlabel('Similarity Threshold', fontsize=16)
        ax1.set_ylabel('Accuracy', fontsize=16)
        ax1.set_title(f'Accuracy vs Similarity Threshold - {task_name.title()} (Full Range)', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Zoomed range (0.95 - 1.0)
        for model, _ in model_performance_at_1_0:
            model_data = task_data[task_data['model'] == model]
            accuracies_zoom = []

            for thresh in thresholds_zoom:
                accuracy = (model_data['similarity'] >= thresh).mean()
                accuracies_zoom.append(accuracy)

            clean_name = clean_model_name(model)
            color = color_map.get(model, None)
            ax2.plot(thresholds_zoom, accuracies_zoom, marker='o', markersize=4, label=clean_name, alpha=0.7, linewidth=2.5, color=color)

        ax2.set_xlabel('Similarity Threshold', fontsize=16)
        ax2.set_ylabel('Accuracy', fontsize=16)
        ax2.set_title(f'Accuracy vs Similarity Threshold - {task_name.title()} (Zoomed: 0.95-1.0)', fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.95, 1.0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Split data by task
    encode_data = df[df['task'] == 'encode']
    decode_data = df[df['task'] == 'decode']

    # Create separate plots
    print("  Encode tasks...")
    create_threshold_sweep_subplot(encode_data, 'encode', encode_save_path, df)
    print("  Decode tasks...")
    create_threshold_sweep_subplot(decode_data, 'decode', decode_save_path, df)


def plot_similarity_distributions(df: pd.DataFrame, save_path: str = None):
    """Plot similarity score distributions for each model."""
    plt.figure(figsize=(15, 10))

    # Get consistent model ordering
    model_performance_ordering = get_model_performance_ordering(df)
    models = [model for model, _ in model_performance_ordering]
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    for i, model in enumerate(models, 1):
        plt.subplot(rows, cols, i)
        model_data = df[df['model'] == model]

        plt.hist(model_data['similarity'], bins=50, alpha=0.7, density=True)
        plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
        plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='100% threshold')

        plt.title(f'{clean_model_name(model)}\n(n={len(model_data)})', fontsize=15)
        plt.xlabel('Similarity Score', fontsize=13)
        plt.ylabel('Density', fontsize=13)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xlim(0, 1)

        if i == 1:  # Add legend to first plot
            plt.legend(fontsize=12)

    plt.suptitle('Similarity Score Distributions by Model', fontsize=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_by_data_type(df: pd.DataFrame, threshold: float = 0.95, save_path: str = None):
    """Plot model performance broken down by data type."""
    # Calculate accuracy by model and data type
    accuracy_by_type = df.groupby(['model', 'data_type'])['similarity'].apply(
        lambda x: (x >= threshold).mean()
    ).reset_index()
    accuracy_by_type.columns = ['model', 'data_type', 'accuracy']

    # Pivot for heatmap
    heatmap_data = accuracy_by_type.pivot(index='data_type', columns='model', values='accuracy')

    # Get consistent model ordering and reorder columns
    model_performance_ordering = get_model_performance_ordering(df)
    ordered_models = [model for model, _ in model_performance_ordering]
    # Keep only models that exist in the heatmap data
    ordered_models = [model for model in ordered_models if model in heatmap_data.columns]
    heatmap_data = heatmap_data[ordered_models]

    # Order data types (rows) by average accuracy across models (best to worst)
    # Best = highest average accuracy (easiest), worst = lowest average accuracy (hardest)
    data_type_avg_accuracy = heatmap_data.mean(axis=1).sort_values(ascending=False)
    heatmap_data = heatmap_data.loc[data_type_avg_accuracy.index]

    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f',
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    plt.title(f'Accuracy by Data Type and Model (Threshold: {threshold})')
    plt.ylabel('Data Type')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_by_data_type_by_task(df: pd.DataFrame, threshold: float = 1.0,
                                         encode_save_path: str = None, decode_save_path: str = None):
    """Plot separate performance by data type heatmaps for encode and decode tasks."""

    def create_performance_heatmap(task_data, task_name, save_path, overall_df):
        """Create a performance by data type heatmap for a specific task."""
        # Calculate accuracy by model and data type for this task
        accuracy_by_type = task_data.groupby(['model', 'data_type'])['similarity'].apply(
            lambda x: (x >= threshold).mean()
        ).reset_index()
        accuracy_by_type.columns = ['model', 'data_type', 'accuracy']

        # Pivot for heatmap
        heatmap_data = accuracy_by_type.pivot(index='data_type', columns='model', values='accuracy')

        # Get task-specific model ordering (based on performance at this task)
        model_performance_ordering = get_model_performance_ordering(task_data)
        ordered_models = [model for model, _ in model_performance_ordering]
        # Keep only models that exist in the heatmap data
        ordered_models = [model for model in ordered_models if model in heatmap_data.columns]
        heatmap_data = heatmap_data[ordered_models]

        # Order data types (rows) by average accuracy across models (best to worst)
        # Best = highest average accuracy (easiest), worst = lowest average accuracy (hardest)
        data_type_avg_accuracy = heatmap_data.mean(axis=1).sort_values(ascending=False)
        heatmap_data = heatmap_data.loc[data_type_avg_accuracy.index]

        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f',
                    cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
        plt.title(f'Accuracy by Data Type and Model - {task_name.title()} (Threshold: {threshold})')
        plt.ylabel('Data Type')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Split data by task
    encode_data = df[df['task'] == 'encode']
    decode_data = df[df['task'] == 'decode']

    # Create separate heatmaps
    print("  Encode tasks...")
    create_performance_heatmap(encode_data, 'encode', encode_save_path, df)
    print("  Decode tasks...")
    create_performance_heatmap(decode_data, 'decode', decode_save_path, df)


def plot_similarity_distributions_by_task(df: pd.DataFrame,
                                         encode_save_path: str = None, decode_save_path: str = None):
    """Plot separate similarity distribution graphs for encode and decode tasks."""

    def create_similarity_distributions(task_data, task_name, save_path, overall_df):
        """Create similarity distribution plots for a specific task."""
        plt.figure(figsize=(15, 10))

        # Get task-specific model ordering (based on performance at this task)
        model_performance_ordering = get_model_performance_ordering(task_data)
        models = [model for model, _ in model_performance_ordering]
        n_models = len(models)
        cols = 3
        rows = (n_models + cols - 1) // cols

        for i, model in enumerate(models, 1):
            plt.subplot(rows, cols, i)
            model_data = task_data[task_data['model'] == model]

            plt.hist(model_data['similarity'], bins=50, alpha=0.7, density=True)
            plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
            plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='100% threshold')

            plt.title(f'{clean_model_name(model)}\n(n={len(model_data)})', fontsize=15)
            plt.xlabel('Similarity Score', fontsize=13)
            plt.ylabel('Density', fontsize=13)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.xlim(0, 1)

            if i == 1:  # Add legend to first plot
                plt.legend(fontsize=12)

        plt.suptitle(f'Similarity Score Distributions by Model - {task_name.title()} Tasks', fontsize=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Split data by task
    encode_data = df[df['task'] == 'encode']
    decode_data = df[df['task'] == 'decode']

    # Create separate distributions
    print("  Encode tasks...")
    create_similarity_distributions(encode_data, 'encode', encode_save_path, df)
    print("  Decode tasks...")
    create_similarity_distributions(decode_data, 'decode', decode_save_path, df)


def plot_performance_by_task(df: pd.DataFrame, threshold: float = 0.95, save_path: str = None):
    """Plot model performance by encoding vs decoding task."""
    accuracy_by_task = df.groupby(['model', 'task'])['similarity'].apply(
        lambda x: (x >= threshold).mean()
    ).reset_index()
    accuracy_by_task.columns = ['model', 'task', 'accuracy']

    # Get consistent model ordering
    model_performance_ordering = get_model_performance_ordering(df)
    ordered_models = [model for model, _ in model_performance_ordering]

    plt.figure(figsize=(15, 8))

    # Filter and order data
    encode_data = accuracy_by_task[accuracy_by_task['task'] == 'encode']
    decode_data = accuracy_by_task[accuracy_by_task['task'] == 'decode']

    # Filter data to only include models in our ordered list and sort by the ordering
    encode_data = encode_data[encode_data['model'].isin(ordered_models)]
    decode_data = decode_data[decode_data['model'].isin(ordered_models)]

    # Create mapping for consistent ordering
    model_order = {model: i for i, model in enumerate(ordered_models)}
    encode_data['order'] = encode_data['model'].map(model_order)
    decode_data['order'] = decode_data['model'].map(model_order)

    encode_data = encode_data.sort_values('order')
    decode_data = decode_data.sort_values('order')

    # Clean model names
    encode_data['clean_model'] = encode_data['model'].apply(clean_model_name)
    decode_data['clean_model'] = decode_data['model'].apply(clean_model_name)

    x = np.arange(len(encode_data))
    width = 0.35

    plt.bar(x - width/2, encode_data['accuracy'], width, label='Encode', alpha=0.7)
    plt.bar(x + width/2, decode_data['accuracy'], width, label='Decode', alpha=0.7)

    plt.xlabel('Model', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title(f'Encoding vs Decoding Performance (Threshold: {threshold})', fontsize=18)
    plt.xticks(x, encode_data['clean_model'], rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_table(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Generate a summary table of model performance."""
    summary = df.groupby('model').agg({
        'similarity': ['count', 'mean', 'std', 'min', 'max'],
        'passed_at_1_0': 'mean',
        'passed_at_95': 'mean',
        'passed_at_90': 'mean',
        'input_length': 'mean',
        'distance': 'mean'
    }).round(4)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.rename(columns={
        'similarity_count': 'total_samples',
        'similarity_mean': 'avg_similarity',
        'similarity_std': 'std_similarity',
        'similarity_min': 'min_similarity',
        'similarity_max': 'max_similarity',
        'passed_at_1_0_mean': 'accuracy_100',
        'passed_at_95_mean': 'accuracy_95',
        'passed_at_90_mean': 'accuracy_90',
        'input_length_mean': 'avg_input_length',
        'distance_mean': 'avg_levenshtein_distance'
    })

    # Sort by accuracy at perfect threshold (1.0)
    summary = summary.sort_values('accuracy_100', ascending=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze base64 benchmark results')
    parser.add_argument('--logs-dir', default='base64bench-logs/results',
                       help='Directory containing eval logs')
    parser.add_argument('--output-dir', default='analysis_plots',
                       help='Directory to save plots')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Primary threshold for analysis')

    args = parser.parse_args()

    print("Loading evaluation results...")
    results = load_eval_results(args.logs_dir)

    print(f"Found results for {len(results)} models:")
    for model, samples in results.items():
        print(f"  {model}: {len(samples)} samples")

    print("\nExtracting similarity scores...")
    df = extract_similarity_scores(results)

    if df.empty:
        print("No data found! Check that eval files contain similarity metadata.")
        return

    print(f"Extracted {len(df)} samples across {df['model'].nunique()} models")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create consistent color map for all models based on overall performance
    print("\nCreating consistent color mapping...")
    color_map = get_model_color_map(df)

    # Generate analyses
    print("\n1. Generating threshold sweep analysis...")
    plot_threshold_sweep(df, os.path.join(args.output_dir, "threshold_sweep.png"), color_map)

    print("\n2. Generating task-specific threshold sweep analysis...")
    plot_threshold_sweep_by_task(df,
                                 os.path.join(args.output_dir, "threshold_sweep_encode.png"),
                                 os.path.join(args.output_dir, "threshold_sweep_decode.png"),
                                 color_map)

    print("\n3. Generating similarity distributions...")
    plot_similarity_distributions(df, os.path.join(args.output_dir, "similarity_distributions.png"))

    print("\n4. Generating task-specific similarity distributions...")
    plot_similarity_distributions_by_task(df,
                                         os.path.join(args.output_dir, "similarity_distributions_encode.png"),
                                         os.path.join(args.output_dir, "similarity_distributions_decode.png"))

    print("\n5. Analyzing performance by data type...")
    plot_performance_by_data_type(df, 1.0, os.path.join(args.output_dir, "performance_by_type.png"))

    print("\n6. Analyzing task-specific performance by data type...")
    plot_performance_by_data_type_by_task(df, 1.0,
                                         os.path.join(args.output_dir, "performance_by_type_encode.png"),
                                         os.path.join(args.output_dir, "performance_by_type_decode.png"))

    print("\n7. Analyzing encoding vs decoding performance...")
    plot_performance_by_task(df, 1.0, os.path.join(args.output_dir, "encode_vs_decode.png"))

    print("8. Generating summary table...")
    summary = generate_summary_table(df, args.threshold)
    summary_path = os.path.join(args.output_dir, "model_summary.csv")
    summary.to_csv(summary_path)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(summary[['total_samples', 'avg_similarity', 'accuracy_100', 'accuracy_95', 'accuracy_90']].to_string())

    print(f"\nAll plots and summary saved to: {args.output_dir}/")
    print("\nFiles generated:")
    print("  - threshold_sweep.png: Accuracy vs threshold curves (combined)")
    print("  - threshold_sweep_encode.png: Accuracy vs threshold curves (encode tasks only)")
    print("  - threshold_sweep_decode.png: Accuracy vs threshold curves (decode tasks only)")
    print("  - similarity_distributions.png: Histogram of similarity scores (combined)")
    print("  - similarity_distributions_encode.png: Histogram of similarity scores (encode tasks only)")
    print("  - similarity_distributions_decode.png: Histogram of similarity scores (decode tasks only)")
    print("  - performance_by_type.png: Heatmap of accuracy by data type (combined, threshold=1.0)")
    print("  - performance_by_type_encode.png: Heatmap of accuracy by data type (encode tasks only, threshold=1.0)")
    print("  - performance_by_type_decode.png: Heatmap of accuracy by data type (decode tasks only, threshold=1.0)")
    print("  - encode_vs_decode.png: Encoding vs decoding performance (threshold=1.0)")
    print("  - model_summary.csv: Detailed performance statistics")


if __name__ == '__main__':
    # Add required imports to requirements if not already present
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Install with: pip install matplotlib seaborn pandas")
        exit(1)

    main()
