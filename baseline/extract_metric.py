import json
import argparse
from pathlib import Path
from utils.run_cmd import relative

def extract_metric_values(input_file, output_file):
    """Extract metric values from a single input file and pad to 128."""
    metric_values = []
    with open(input_file, 'r') as f:
        # Loop through lines
        for line in f:
            # Split line by comma
            bundle_name, value = line.strip().split(',')
            metric_values.append(float(value))

    # Zero pad to 128
    metric_values += [0] * (128 - len(metric_values))
    
    with open(output_file, 'w') as f:
        json.dump(metric_values, f, indent=4)
    
    print(f"Extracted values saved to {relative(output_file)}")

def extract_multiple_metrics(metric_files, output_file, metric_order=['fa', 'md', 'ad', 'rd'], pad_to=-1):
    """
    Extract and combine multiple metrics into a single 512-element vector.
    
    Args:
        metric_files: Dictionary mapping metric names to file paths
                     e.g., {'fa': 'path/to/fa.json', 'md': 'path/to/md.json', ...}
        output_file: Path to output JSON file
    """
    all_metrics = []
    feature_size = 72
    for metric_name in metric_order:
        if metric_name in metric_files:
            metric_file = metric_files[metric_name]
            metric_values = []
            
            if Path(metric_file).exists():
                with open(metric_file, 'r') as f:
                    for line in f:
                        bundle_name, value = line.strip().split(',')
                        metric_values.append(float(value))
            else:
                print(f"Warning: {relative(metric_file)} does not exist, using zeros for {metric_name}")

            if pad_to != -1:
                metric_values += [0] * (pad_to - len(metric_values))
            all_metrics.extend(metric_values)
        else:
            # If metric file not provided, add 128 zeros
            print(f"Warning: {metric_name} not provided, using zeros")
            all_metrics.extend([0] * feature_size)
    
    # Ensure total length is 512 (4 metrics * 128 each)
    total_length = len(metric_order) * feature_size
    if len(all_metrics) < total_length:
        all_metrics += [0] * (total_length - len(all_metrics))
    elif len(all_metrics) > total_length:
        all_metrics = all_metrics[:total_length]
    
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"Extracted {len(all_metrics)} values (combined metrics) saved to {relative(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mean metric values from scilpy JSON file(s).")
    parser.add_argument("--fa", help="Path to FA metric file (bundle_name,value format)")
    parser.add_argument("--md", help="Path to MD metric file (bundle_name,value format)")
    parser.add_argument("--ad", help="Path to AD metric file (bundle_name,value format)")
    parser.add_argument("--rd", help="Path to RD metric file (bundle_name,value format)")
    parser.add_argument("input_file", nargs='?', help="Path to input file bundle_name,value (for single metric)")
    parser.add_argument("output_file", nargs='?', help="Path to output file (list of mean metric values in ROIs)")
    
    args = parser.parse_args()
    
    # If multiple metrics provided, use extract_multiple_metrics
    if args.fa or args.md or args.ad or args.rd:
        metric_files = {}
        if args.fa:
            metric_files['fa'] = args.fa
        if args.md:
            metric_files['md'] = args.md
        if args.ad:
            metric_files['ad'] = args.ad
        if args.rd:
            metric_files['rd'] = args.rd
        
        if not args.output_file:
            parser.error("output_file is required when using multiple metrics")
        extract_multiple_metrics(metric_files, args.output_file)
    else:
        # Single metric mode (backward compatible)
        if not args.input_file or not args.output_file:
            parser.error("input_file and output_file are required for single metric mode")
        extract_metric_values(args.input_file, args.output_file)
