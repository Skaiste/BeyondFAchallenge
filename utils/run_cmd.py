import os
import subprocess
from pathlib import Path


def run_command(cmd, check=True, print_output=True):
    """Run a shell command and return the result, printing stdout/stderr in real-time.
    
    Args:
        cmd: Command to run as a list of strings
        check: If True, raise CalledProcessError on non-zero exit code
        print_output: If True (default), print stdout/stderr in real-time
    """
    if print_output:
        print(f"Running: {' '.join(cmd)}")
    try:
        # Special handling for TractSeg to avoid OpenMP conflicts
        env = None
        if cmd[0] == "TractSeg":
            # Create a clean environment for TractSeg
            # Limit OpenMP threads to reduce conflicts
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            # Unset problematic OpenMP variables that might cause conflicts
            env.pop("KMP_DUPLICATE_LIB_OK", None)
            env.pop("DYLD_INSERT_LIBRARIES", None)
            # Try to use a single OpenMP runtime by limiting thread usage
            print("  Note: Running TractSeg with OMP_NUM_THREADS=1 to avoid OpenMP conflicts")
        
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            env=env,
            bufsize=1  # Line buffered
        )
        
        # Collect output while printing in real-time
        stdout_lines = []
        stderr_lines = []
        
        # Read stdout and stderr line by line using threads for real-time output
        import sys
        import re
        from threading import Thread
        
        def is_progress_bar_line(line):
            """Detect if a line is a progress bar update (tqdm-style)."""
            # Check for carriage return (common in progress bars)
            if line.endswith('\r'):
                return True
            # Check for progress bar patterns: percentage, bars, it/s
            stripped = line.strip()
            # Match patterns like: "  0%|          | 0/144 [00:00<?, ?it/s]"
            # or "   1%|▏         | 2/144 [00:00<00:34,  4.17it/s]"
            if re.search(r'\d+%\|.*\|.*\[.*it/s\]', stripped):
                return True
            # Also check for lines with progress bar characters and percentage
            if re.search(r'\d+%', stripped) and ('|' in stripped or 'it/s' in stripped):
                # Additional check: progress bars are usually short and contain specific chars
                if len(stripped) < 150 and ('█' in stripped or '▊' in stripped or '▉' in stripped or 
                                            '▋' in stripped or '▌' in stripped or '▍' in stripped or
                                            '▎' in stripped or '▏' in stripped):
                    return True
            return False
        
        def read_stdout():
            last_was_progress = False
            for line in iter(process.stdout.readline, ''):
                if line:
                    is_progress = is_progress_bar_line(line)
                    if print_output:
                        if is_progress:
                            # Overwrite the same line for progress bars
                            print(f"\r  {line.rstrip()}", end='', flush=True)
                            last_was_progress = True
                        else:
                            # If previous line was progress, add newline first
                            if last_was_progress:
                                print()  # Newline after progress bar
                                last_was_progress = False
                            print(f"  {line.rstrip()}")
                    stdout_lines.append(line)
            # If last line was progress, add final newline
            if print_output and last_was_progress:
                print()
            process.stdout.close()
        
        def read_stderr():
            last_was_progress = False
            for line in iter(process.stderr.readline, ''):
                if line:
                    is_progress = is_progress_bar_line(line)
                    if print_output:
                        if is_progress:
                            # Overwrite the same line for progress bars
                            print(f"\r  {line.rstrip()}", end='', file=sys.stderr, flush=True)
                            last_was_progress = True
                        else:
                            # If previous line was progress, add newline first
                            if last_was_progress:
                                print(file=sys.stderr)  # Newline after progress bar
                                last_was_progress = False
                            print(f"  {line.rstrip()}", file=sys.stderr)
                    stderr_lines.append(line)
            # If last line was progress, add final newline
            if print_output and last_was_progress:
                print(file=sys.stderr)
            process.stderr.close()
        
        # Start threads to read stdout and stderr concurrently
        stdout_thread = Thread(target=read_stdout)
        stderr_thread = Thread(target=read_stderr)
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        returncode = process.wait()
        
        # Wait for threads to finish reading
        stdout_thread.join()
        stderr_thread.join()
        
        # Create a result object similar to subprocess.run
        class Result:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        result = Result(returncode, ''.join(stdout_lines), ''.join(stderr_lines))
        
        # If command failed but check=False, print error info
        if result.returncode != 0 and not check and print_output:
            print(f"  WARNING: Command exited with code {result.returncode}")
        
        # Check for errors if check=True
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        return result
        
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Command not found: {cmd[0]}")
        print(f"{'='*60}")
        print(f"  Full error: {e}")
        print(f"  Make sure {cmd[0]} is installed and in your PATH")
        print(f"  Current PATH: {os.environ.get('PATH', 'Not set')[:200]}...")
        print(f"{'='*60}\n")
        raise
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"{'='*60}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Return code: {e.returncode}")
        
        if e.stdout:
            print(f"\n  stdout:\n{e.stdout}")
        else:
            print(f"\n  stdout: (empty)")
            
        if e.stderr:
            print(f"\n  stderr:\n{e.stderr}")
        else:
            print(f"\n  stderr: (empty)")
            
        print(f"{'='*60}\n")
        raise

PROJECT_ROOT = None
def get_project_root():
    """
    Get the project root directory.
    """
    global PROJECT_ROOT
    if PROJECT_ROOT is None:
        # Start from this file's directory and walk up for .git
        start_path = Path(__file__).resolve().parent
        cur = start_path
        PROJECT_ROOT = None
        for parent in [cur] + list(cur.parents):
            if (parent / ".git").is_dir():
                PROJECT_ROOT = parent
                break
        if PROJECT_ROOT is None:
            # Default: use start_path if .git not found
            PROJECT_ROOT = start_path
    return PROJECT_ROOT

def relative(path):
    """
    Convert an absolute path to a path relative to the project root directory.

    The project root is assumed to be the first parent directory (of this file)
    that contains a .git directory, or the top-level containing this script if not found.
    """
    project_root = get_project_root()
    path = Path(path).resolve()
    try:
        rel_path = path.relative_to(project_root)
    except ValueError:
        # If path is not under project_root, just return as-is
        return str(path)
    return str(rel_path)
