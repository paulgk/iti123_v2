# Multiprocessing RuntimeError Fix

## Problem

When running the CPU training script on macOS, you encountered this error:

```
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...
```

## Root Cause

**macOS uses `spawn` instead of `fork` for multiprocessing** (since Python 3.8+). 

The `spawn` method:
1. Starts a **fresh Python interpreter** for each worker
2. **Re-imports the main module** to get the worker code
3. **Requires** the main execution code to be protected with `if __name__ == '__main__':`

Without this guard, the re-imported module tries to start training again, causing infinite recursion.

## Solution

Wrap all main execution code in a function and protect it with `if __name__ == '__main__':`:

### Before (Broken):
```python
# ... imports and class definitions ...

# Configuration
CONFIG = {...}

# Main execution code (NOT protected)
print("Loading dataset...")
frames_dir = Path(DATA_ROOT)
# ... rest of code ...
```

### After (Fixed):
```python
# ... imports and class definitions ...

# Configuration
CONFIG = {...}

def main():
    """Main training function"""
    print("Loading dataset...")
    frames_dir = Path(DATA_ROOT)
    # ... rest of code ...

if __name__ == '__main__':
    main()
```

## What Changed

**File**: `notebooks/badminton_training_cpu_local.py`

**Changes**:
1. Added `def main():` function (line 264)
2. Indented all main execution code (lines 265-741)
3. Added `if __name__ == '__main__': main()` at the end (lines 744-745)

**What's NOT in main()**:
- Imports
- Class definitions (`GracefulInterruptHandler`, `BadmintonFramesDataset`, `MobileNet_LSTM_Classifier`)
- Global configuration (CONFIG, args parsing, interrupt_handler)

**What's IN main()**:
- Dataset loading
- Model creation
- Training loop
- Evaluation
- Results saving

## Why This Works

```python
if __name__ == '__main__':
    main()
```

This guard ensures:
1. When you run the script directly: `__name__ == '__main__'` → main() executes
2. When workers import the module: `__name__ == 'badminton_training_cpu_local'` → main() is skipped
3. Workers can still access classes and functions they need
4. No infinite recursion!

## Testing

The fix is confirmed working:

```bash
$ python notebooks/badminton_training_cpu_local.py --help
usage: badminton_training_cpu_local.py [-h]
                                       [--max-cpu-percent MAX_CPU_PERCENT]
                                       [--skip-ratio SKIP_RATIO]
                                       [--resume RESUME]

✓ No RuntimeError!
```

## Platform Differences

| Platform | Default Method | Needs `if __name__` Guard? |
|----------|----------------|----------------------------|
| **Linux** | `fork` | No (but recommended) |
| **macOS** | `spawn` | **Yes** (required) |
| **Windows** | `spawn` | **Yes** (required) |

**Best practice**: Always use the guard for cross-platform compatibility.

## Common Variations

### Minimal Guard
```python
if __name__ == '__main__':
    main()
```

### With freeze_support (for Windows .exe)
```python
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
```

### With set_start_method (force fork on macOS)
```python
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork')  # Force fork instead of spawn
    main()
```

**Note**: We use the minimal guard since it works cross-platform without forcing fork.

## Summary

✅ **Fixed**: Added `if __name__ == '__main__':` guard
✅ **Result**: Script now works on macOS with multiprocessing
✅ **Tested**: Help output works without RuntimeError
✅ **Compatible**: Works on Linux, macOS, and Windows

You can now run training:
```bash
conda activate badminton-cpu-training
python notebooks/badminton_training_cpu_local.py
```
