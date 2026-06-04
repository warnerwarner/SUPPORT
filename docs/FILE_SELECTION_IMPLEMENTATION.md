# File Selection and Filtering Implementation

## Overview
Added intelligent file selection and filtering capabilities to SUPPORT training scripts. The system now:
1. Recursively searches for `.tif` files only (no more `.png` or other files)
2. Filters files based on `--is_raw` flag (raw vs reconstructed data)
3. Randomly selects a subset if `--training_size` is specified
4. Uses `--random_seed` for reproducible sampling

## Changes Implemented

### 1. Fixed Critical Bug in `train_distributed.py`
**Location:** Line 195-199

**REMOVED hardcoded path:**
```python
data_dir = "/gpfs/home/warnet02/data/stephen/run012"  # ❌ REMOVED
noisy_data = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]  # ❌ REMOVED
opt.noisy_data = noisy_data  # ❌ REMOVED
```

**REPLACED with:**
```python
# Use the noisy_data list already processed by util.parse_arguments()
noisy_data = opt.noisy_data  # ✅ Now respects command-line arguments!
```

**Impact:** Training now uses the files specified in command-line arguments instead of ignoring them.

---

### 2. Added `--training_size` Argument
**Files Modified:**
- `src/utils/util.py` (both `parse_arguments()` and `get_opts()`)

**New parameter:**
```python
parser.add_argument("--training_size", type=int, default=None, 
                    help="number of .tif files to randomly select for training (default: use all)")
```

**Usage:**
```bash
--training_size 50  # Randomly select 50 files
```

---

### 3. Set Explicit Default for `--is_raw`
**Files Modified:**
- `src/utils/util.py` (both parsers)

**Updated:**
```python
parser.add_argument("--is_raw", action="store_true", default=False, 
                    help="use anisotropic dilations for raw voltage imaging data (Y minimal, X exponential). Default: False (uses processed/reconstructed data)")
```

**Default behavior:** When `--is_raw` is NOT specified, assumes processed/reconstructed data.

---

### 4. Implemented Smart File Selection Logic
**Files Modified:**
- `src/utils/util.py` (`parse_arguments()` lines 64-125, `update_opt()` lines 182-241)

**New behavior:**

#### A. **File Type Filtering**
- Only searches for `*.tif` files (was `*` before)
- Ignores `.png`, `.txt`, and other non-TIF files

#### B. **Raw vs Reconstructed Filtering**
```python
if opt.is_raw:
    # Exclude files with 'reconstructed' in filename
    tif_files = [f for f in tif_files if 'reconstructed' not in Path(f).name]
else:
    # Include ONLY files with 'reconstructed' in filename
    tif_files = [f for f in tif_files if 'reconstructed' in Path(f).name]
```

#### C. **Random Subset Selection**
```python
if opt.training_size is not None:
    if opt.training_size < len(all_files):
        random.seed(opt.random_seed)  # Reproducible
        all_files = random.sample(all_files, opt.training_size)
        all_files = sorted(all_files)  # Re-sort for consistency
    else:
        print(f"WARNING: Requested {opt.training_size} files, but only {len(all_files)} available. Using all {len(all_files)} files.")
```

#### D. **Informative Logging**
```python
print("=" * 70)
print("FILE SELECTION SUMMARY")
print("=" * 70)
filter_type = "raw (excluding 'reconstructed')" if opt.is_raw else "processed (only 'reconstructed')"
print(f"Filter mode: {filter_type}")
if opt.training_size is not None:
    print(f"Requested training size: {opt.training_size} files")
print(f"Total files selected: {len(opt.noisy_data)}")
print("=" * 70)
```

---

## Usage Examples

### Example 1: Raw voltage data, 50 random files
```bash
python -m src.train_distributed \
    --is_folder \
    --noisy_data /gpfs/home/warnet02/data/stephen \
    --is_raw \
    --training_size 50 \
    --patch_size 61 16 320 \
    --depth 8 \
    --bs_size 1 3
```

**Result:**
- Searches recursively in `/data/stephen` for `*.tif` files
- Excludes files with "reconstructed" in name → ~550 raw files found
- Randomly selects 50 files (using `random_seed=0` for reproducibility)
- Uses anisotropic dilations (`is_raw=True`)

---

### Example 2: Processed data, all available files from run012
```bash
python -m src.train_distributed \
    --is_folder \
    --noisy_data /gpfs/home/warnet02/data/stephen/run012 \
    --patch_size 61 16 320
```
(Note: No `--is_raw` flag, no `--training_size`)

**Result:**
- Searches in `run012` for `*.tif` files
- Includes ONLY files with "reconstructed" in name → ~80 files found
- Uses all 80 files (no random sampling)
- Uses isotropic dilations (`is_raw=False` by default)

---

### Example 3: Request more files than available
```bash
python -m src.train_distributed \
    --is_folder \
    --noisy_data /gpfs/home/warnet02/data/stephen/run012 \
    --is_raw \
    --training_size 1000
```

**Result:**
- Finds ~80 raw files
- Prints: `WARNING: Requested 1000 files, but only 80 available. Using all 80 files.`
- Uses all 80 available files

---

## File Selection Summary Output

When running training, you'll see:
```
======================================================================
FILE SELECTION SUMMARY
======================================================================
Filter mode: raw (excluding 'reconstructed')
Requested training size: 50 files
Total files selected: 50
======================================================================

Noisy files:
/gpfs/home/warnet02/data/stephen/run012/stim_v1_fov4_440Hz_..._00006.tif
/gpfs/home/warnet02/data/stephen/run012/stim_v1_fov4_440Hz_..._00034.tif
...
```

---

## Testing Results

✅ **Test 1:** Raw data, 5 files from run012
- Filter mode: `raw (excluding 'reconstructed')`
- Files selected: 5 (randomly sampled)
- No "reconstructed" files in selection ✓

✅ **Test 2:** Processed data, all files from run012
- Filter mode: `processed (only 'reconstructed')`
- Files selected: 80 (all available reconstructed files)
- Only "reconstructed" files in selection ✓

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `src/train_distributed.py` | 195-199 | Removed hardcoded path, use `opt.noisy_data` |
| `src/utils/util.py` | 25 | Added `--training_size` argument (`parse_arguments`) |
| `src/utils/util.py` | 109 | Added `--training_size` argument (`get_opts`) |
| `src/utils/util.py` | 34, 118 | Made `--is_raw` default explicit (`False`) |
| `src/utils/util.py` | 64-125 | Implemented smart file selection (`parse_arguments`) |
| `src/utils/util.py` | 182-241 | Implemented smart file selection (`update_opt`) |

---

## Key Features

1. ✅ **Recursive search** - Finds files in all subdirectories
2. ✅ **Type filtering** - Only `.tif` files
3. ✅ **Content filtering** - Raw vs reconstructed based on filename
4. ✅ **Random sampling** - Reproducible with `random_seed`
5. ✅ **Warning messages** - Alerts when fewer files available than requested
6. ✅ **Informative output** - Clear summary of selection criteria and results
7. ✅ **Backward compatible** - Default behavior matches expected use case

---

## Benefits

### For Training Efficiency:
- **Quick iteration:** Test architecture on 50-100 files instead of all 550
- **Resource management:** Reduce data loading and memory requirements
- **Parallel experiments:** Different random seeds → different training sets

### For Data Quality:
- **No accidental mixing:** Clear separation between raw and reconstructed data
- **Reproducibility:** Same seed → same file selection every time
- **Validation:** Easy to verify correct data is being used

### For Development:
- **Fixed critical bug:** Training now respects command-line arguments
- **Better UX:** Clear feedback on what files are being used
- **Flexible:** Easy to switch between raw/processed or subset/full training
