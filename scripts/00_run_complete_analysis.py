"""
Complete Analysis Runner with Comprehensive Logging

This script runs the entire analysis pipeline and logs all outputs to a file.
It captures:
- All print statements
- Error messages
- Execution times
- Summary statistics
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import traceback
import time
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Create logs directory
LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)

# Create timestamp for log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOGS_DIR / f'complete_analysis_{timestamp}.log'
summary_file = LOGS_DIR / f'analysis_summary_{timestamp}.txt'

class TeeOutput:
    """Class to write to both file and console simultaneously"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, text):
        # Write to file with full Unicode support
        self.file.write(text)
        # Write to console with ASCII fallback for Windows compatibility
        try:
            self.stdout.write(text)
        except UnicodeEncodeError:
            # Replace Unicode characters with ASCII equivalents for console
            ascii_text = text.encode('ascii', 'replace').decode('ascii')
            self.stdout.write(ascii_text)
        self.file.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()

def log_section(tee, title, char='='):
    """Log a section header"""
    tee.write(f"\n{char * 80}\n")
    tee.write(f"{title}\n")
    tee.write(f"{char * 80}\n\n")

def run_with_logging(tee, step_name, func, *args, **kwargs):
    """Run a function and log its output"""
    start_time = time.time()
    log_section(tee, f"STEP: {step_name}", '=')
    tee.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    success = False
    error_msg = None
    
    try:
        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = func(*args, **kwargs)
        
        # Write captured output
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        if stdout_text:
            tee.write(stdout_text)
        if stderr_text:
            tee.write(f"STDERR:\n{stderr_text}\n")
        
        elapsed_time = time.time() - start_time
        tee.write(f"\n✓ {step_name} completed successfully\n")
        tee.write(f"Execution time: {elapsed_time:.2f} seconds\n")
        success = True
        
        return result, success, None, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        tee.write(f"\n✗ ERROR in {step_name}:\n")
        tee.write(f"{error_msg}\n\n")
        tee.write(f"Traceback:\n{error_trace}\n")
        tee.write(f"Execution time: {elapsed_time:.2f} seconds\n")
        
        return None, False, error_msg, elapsed_time

def run_data_collection(tee):
    """Run data collection"""
    try:
        # Import here to catch any import errors
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        # Try new data collection first, fallback to old
        try:
            # Import from scripts directory
            import importlib.util
            spec = importlib.util.spec_from_file_location("data_collection", Path(__file__).parent / "01_data_collection.py")
            data_collection = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_collection)
            collect_main = data_collection.main
        except ImportError:
            from data_collection import main as collect_main
        return run_with_logging(tee, "Data Collection", collect_main)
    except ImportError as e:
        tee.write(f"✗ Could not import data_collection: {e}\n")
        tee.write(f"  Traceback:\n{traceback.format_exc()}\n")
        return None, False, str(e), 0
    except Exception as e:
        tee.write(f"✗ Unexpected error in data collection: {e}\n")
        tee.write(f"  Traceback:\n{traceback.format_exc()}\n")
        return None, False, str(e), 0

def run_eda(tee):
    """Run exploratory data analysis"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("eda", Path(__file__).parent / "02_exploratory_data_analysis.py")
        eda_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eda_module)
        eda_main = eda_module.main
        return run_with_logging(tee, "Exploratory Data Analysis", eda_main)
    except ImportError as e:
        tee.write(f"✗ Could not import eda: {e}\n")
        return None, False, str(e), 0

def run_hypothesis_testing(tee):
    """Run hypothesis testing"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hypothesis_testing", Path(__file__).parent / "03_hypothesis_testing.py")
        hyp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hyp_module)
        hyp_main = hyp_module.main
        return run_with_logging(tee, "Hypothesis Testing", hyp_main)
    except ImportError as e:
        tee.write(f"✗ Could not import hypothesis_testing: {e}\n")
        return None, False, str(e), 0


def check_outputs(tee):
    """Check what output files were created"""
    log_section(tee, "Output Files Check", '-')
    
    OUTPUT_DIR = Path('outputs')
    PROCESSED_DIR = Path('data/processed')
    
    outputs_found = []
    outputs_missing = []
    
    expected_outputs = [
        ('outputs/eda_report.txt', 'EDA Report'),
        ('outputs/hypothesis_testing_report.txt', 'Hypothesis Testing Report'),
        ('outputs/correlation_heatmap.png', 'Correlation Heatmap'),
        ('outputs/distributions.png', 'Distribution Plots'),
        ('outputs/boxplots.png', 'Box Plots'),
        ('outputs/feature_relationships.png', 'Feature Relationships'),
        ('outputs/categorical_analysis.png', 'Categorical Analysis'),
        ('outputs/hypothesis_1.png', 'Hypothesis 1 Plot'),
        ('outputs/hypothesis_2.png', 'Hypothesis 2 Plot'),
        ('outputs/hypothesis_3.png', 'Hypothesis 3 Plot'),
        ('outputs/hypothesis_4.png', 'Hypothesis 4 Plot'),
        ('outputs/hypothesis_5.png', 'Hypothesis 5 Plot'),
        ('outputs/hypothesis_6.png', 'Hypothesis 6 Plot'),
        ('data/processed/final_combined_dataset.csv', 'Final Combined Dataset'),
        ('data/processed/merged_data_combined.csv', 'Merged Dataset'),
    ]
    
    for file_path, description in expected_outputs:
        full_path = Path(file_path)
        if full_path.exists():
            size = full_path.stat().st_size
            outputs_found.append((description, file_path, size))
            tee.write(f"✓ {description}: {file_path} ({size:,} bytes)\n")
        else:
            outputs_missing.append((description, file_path))
            tee.write(f"✗ {description}: {file_path} (NOT FOUND)\n")
    
    tee.write(f"\nSummary: {len(outputs_found)} files found, {len(outputs_missing)} missing\n")
    
    return outputs_found, outputs_missing

def generate_summary_report(tee, results, summary_file_path):
    """Generate a summary report"""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("COMPLETE ANALYSIS SUMMARY REPORT")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Log File: {log_file}")
    summary_lines.append("")
    
    # Overall status
    all_success = all(result[1] for result in results.values())
    summary_lines.append(f"Overall Status: {'✓ SUCCESS' if all_success else '⚠ PARTIAL SUCCESS'}")
    summary_lines.append("")
    
    # Step results
    summary_lines.append("STEP RESULTS:")
    summary_lines.append("-" * 80)
    
    total_time = 0
    for step_name, (result, success, error, elapsed_time) in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        summary_lines.append(f"{step_name}:")
        summary_lines.append(f"  Status: {status}")
        summary_lines.append(f"  Execution Time: {elapsed_time:.2f} seconds")
        if error:
            summary_lines.append(f"  Error: {error}")
        summary_lines.append("")
        total_time += elapsed_time
    
    summary_lines.append(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    summary_lines.append("")
    
    # Output files
    summary_lines.append("OUTPUT FILES:")
    summary_lines.append("-" * 80)
    
    outputs_found, outputs_missing = check_outputs(tee)
    
    summary_lines.append(f"\nFiles Generated: {len(outputs_found)}")
    summary_lines.append(f"Files Missing: {len(outputs_missing)}")
    
    if outputs_found:
        summary_lines.append("\nGenerated Files:")
        for desc, path, size in outputs_found:
            summary_lines.append(f"  ✓ {desc}: {path} ({size:,} bytes)")
    
    if outputs_missing:
        summary_lines.append("\nMissing Files:")
        for desc, path in outputs_missing:
            summary_lines.append(f"  ✗ {desc}: {path}")
    
    # Recommendations
    summary_lines.append("")
    summary_lines.append("NEXT STEPS:")
    summary_lines.append("-" * 80)
    
    if all_success:
        summary_lines.append("1. Review outputs/ directory for all generated visualizations")
        summary_lines.append("2. Read outputs/eda_report.txt for EDA summary")
        summary_lines.append("3. Read outputs/hypothesis_testing_report.txt for hypothesis results")
        summary_lines.append("4. Check logs/ directory for detailed execution logs")
    else:
        summary_lines.append("1. Review the log file for error details")
        summary_lines.append("2. Check that all dependencies are installed")
        summary_lines.append("3. Verify data files are available in data/raw/")
        summary_lines.append("4. Re-run failed steps individually if needed")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    # Write summary to file
    summary_text = "\n".join(summary_lines)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    # Also write to tee
    tee.write("\n")
    tee.write(summary_text)
    tee.write("\n")
    
    return summary_text

def main():
    """Main execution function"""
    # Setup tee output (write to both file and console)
    tee = TeeOutput(log_file)
    
    # Redirect stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Log header
        log_section(tee, "GRADUATE ADMISSION PREDICTION - COMPLETE ANALYSIS", '=')
        tee.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"Log File: {log_file}\n")
        tee.write(f"Summary File: {summary_file}\n")
        tee.write(f"Python Version: {sys.version}\n")
        tee.write(f"Working Directory: {os.getcwd()}\n\n")
        
        # Store results
        results = {}
        overall_start_time = time.time()
        
        # Step 1: Data Collection
        data_result, success, error, elapsed = run_data_collection(tee)
        results['1. Data Collection'] = (data_result, success, error, elapsed)
        
        # Step 2: EDA
        eda_result, success, error, elapsed = run_eda(tee)
        results['2. Exploratory Data Analysis'] = (None, success, error, elapsed)
        
        # Step 3: Hypothesis Testing
        hyp_result, success, error, elapsed = run_hypothesis_testing(tee)
        results['3. Hypothesis Testing'] = (None, success, error, elapsed)
        
        # Overall timing
        total_elapsed = time.time() - overall_start_time
        
        # Check outputs
        log_section(tee, "Output Verification", '=')
        outputs_found, outputs_missing = check_outputs(tee)
        
        # Generate summary
        log_section(tee, "Generating Summary Report", '=')
        summary_text = generate_summary_report(tee, results, summary_file)
        
        # Final summary
        log_section(tee, "ANALYSIS COMPLETE", '=')
        tee.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"Total Execution Time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)\n\n")
        
        all_success = all(result[1] for result in results.values())
        if all_success:
            tee.write("✓ ALL STEPS COMPLETED SUCCESSFULLY!\n")
        else:
            tee.write("⚠ SOME STEPS HAD ERRORS - CHECK LOG FOR DETAILS\n")
        
        tee.write(f"\nLog file saved to: {log_file}\n")
        tee.write(f"Summary file saved to: {summary_file}\n")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"Log file: {log_file}")
        print(f"Summary file: {summary_file}")
        print(f"Total time: {total_elapsed:.2f} seconds")
        status_text = "SUCCESS" if all_success else "PARTIAL"
        print(f"Status: {status_text}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        tee.write("\n\n✗ Analysis interrupted by user\n")
        tee.write(f"Partial log saved to: {log_file}\n")
    except Exception as e:
        tee.write(f"\n\n✗ FATAL ERROR: {e}\n")
        tee.write(f"Traceback:\n{traceback.format_exc()}\n")
        tee.write(f"Partial log saved to: {log_file}\n")
    finally:
        tee.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

if __name__ == "__main__":
    main()

