from utils.run_cmd import run_command, relative

def bundle_metrics(metric_dir, roi_list, tractseg_output_dir, file_id, metrics=['fa', 'md', 'ad', 'rd']):
    """
    Calculate the metrics for each bundle.
    """
    tensor_metrics_files = {}
    for metric in metrics:
        tensor_metrics_files[metric] = tractseg_output_dir / file_id / f"tensor_metrics_{metric}.json"
        tensor_metrics_files[metric].parent.mkdir(parents=True, exist_ok=True)
    
    # Check if all tensor metrics files already exist
    all_cached = all(f.exists() and f.stat().st_size > 0 for f in tensor_metrics_files.values())
    
    if all_cached:
        print(f"  [CACHE] Tensor metrics files already exist, skipping fslstats calculations")
    else:
        # Process each metric
        for metric in metrics:
            tensor_metrics_file = tensor_metrics_files[metric]
            metric_file = metric_dir / f"{metric}.nii.gz"
            
            # Check if metric file exists
            if not metric_file.exists():
                print(f"  Warning: {relative(metric_file)} does not exist, skipping {metric}")
                # Create empty file
                tensor_metrics_file.write_text("")
                continue
            
            # Clear/create the file
            tensor_metrics_file.write_text("")
            
            if len(roi_list) == 0:
                print(f"  No bundle ROIs found. Creating empty tensor metrics file for {metric}.")
            else:
                print(f"  Processing {metric} in {len(roi_list)} bundles...")
                for roi in roi_list:
                    bundle_name = roi.stem
                    
                    # Check if sum of mask > 0
                    result = run_command([
                        "fslstats", str(roi), "-V"
                    ], print_output=False)
                    mask_sum = int(result.stdout.split()[0])
                    
                    if mask_sum == 0:
                        with tensor_metrics_file.open('a') as f:
                            f.write(f"{bundle_name},0\n")
                    else:
                        result = run_command([
                            "fslstats", str(metric_file),
                            "-k", str(roi), "-m"
                        ], print_output=False)
                        mean_metric = result.stdout.strip()
                        with tensor_metrics_file.open('a') as f:
                            f.write(f"{bundle_name},{mean_metric}\n")

    return tensor_metrics_files