# Complete Documentation Index: --use_phase_conditioning Flag

## Quick Navigation

### For Different Use Cases:

**I have 5 minutes:**
- Read: PHASE_CONDITIONING_VISUAL_GUIDE.txt (flow diagram & architecture comparison)

**I have 15 minutes:**
- Read: PHASE_CONDITIONING_SUMMARY.md (5-stage pipeline, key details)

**I have 30 minutes:**
- Read: PHASE_CONDITIONING_SUMMARY.md + PHASE_CONDITIONING_QUICK_REF.md

**I need to understand everything:**
- Read all documents in this order:
  1. README_PHASE_CONDITIONING.md
  2. PHASE_CONDITIONING_VISUAL_GUIDE.txt
  3. PHASE_CONDITIONING_SUMMARY.md
  4. PHASE_CONDITIONING_QUICK_REF.md
  5. PHASE_CONDITIONING_TRACE.md

**I need to debug an issue:**
- Start: PHASE_CONDITIONING_QUICK_REF.md → "Common Issues & Debugging"
- Then: PHASE_CONDITIONING_SUMMARY.md → "Discovered Issues"
- Finally: PHASE_CONDITIONING_TRACE.md (search by file/line number)

**I need to modify code:**
- Use: PHASE_CONDITIONING_QUICK_REF.md (file locations table + line numbers)
- Reference: PHASE_CONDITIONING_TRACE.md (detailed implementation)

---

## Document Descriptions

### 1. README_PHASE_CONDITIONING.md (7.3 KB)
**Type:** Index and Navigation Guide
**Best For:** Entry point, overview, getting oriented

**Contains:**
- Overview of documentation suite
- Quick start command examples
- Data format changes (with vs without phase)
- Architecture changes summary
- Key tensor shapes table
- Phase extraction method overview
- Requirements checklist
- Known issues summary
- Testing examples with code
- Performance considerations
- References to test files

**Start here if:** You're new to phase conditioning

---

### 2. PHASE_CONDITIONING_VISUAL_GUIDE.txt (5.8 KB)
**Type:** ASCII Diagrams and Flow Charts
**Best For:** Visual learners, quick understanding

**Contains:**
- Complete flow diagram (command → output)
- Architecture differences table (with vs without)
- Tensor shape evolution diagrams
- Critical implementation details
- Why certain design choices were made

**Start here if:** You prefer visual explanations

---

### 3. PHASE_CONDITIONING_SUMMARY.md (11 KB)
**Type:** Executive Summary
**Best For:** Understanding overall flow and key implementation

**Contains:**
- 5-stage pipeline explanation (stages 1-5 detailed)
- Flag definition details
- Data loading and phase extraction flow
- Batch generation format
- Training loop implementation
- Model processing logic
- Key implementation details:
  - Phase tensor shapes at each stage
  - Model architecture adjustments
  - Data augmentation impact
  - Distributed training setup
- Critical code paths (3 execution paths)
- Discovered issues with fixes
- Testing checklist (8 items)

**Start here if:** You want comprehensive overview in 20-30 minutes

---

### 4. PHASE_CONDITIONING_QUICK_REF.md (6.0 KB)
**Type:** Quick Reference
**Best For:** Fast lookup while coding/debugging

**Contains:**
- File locations summary table (23 entries with line numbers)
- Data flow overview diagram
- Key shapes table throughout pipeline
- Conditional logic examples
- Critical dependencies
- Common issues and debugging
- Testing code snippets

**Start here if:** You need to quickly find where something is

---

### 5. PHASE_CONDITIONING_TRACE.md (24 KB)
**Type:** Detailed Technical Reference
**Best For:** Complete understanding, debugging deep issues

**Contains:**
- 15 sections covering all aspects
- Command-line argument definition
- Dataloader creation path
- Phase extraction in dataloader
- Phase extraction functions (CPU and GPU)
- Dataset batch generation
- Training data unpacking
- Model forward pass
- Model initialization
- SUPPORT model architecture details
- Forward pass with phase (detailed)
- Data flow diagram
- Random transform with phase
- Distributed training handling
- Summary table
- Example execution paths
- Key architectural changes

**Start here if:** You need comprehensive understanding with all details

---

## File Cross-Reference

### By Topic:

**Flag Definition:**
- PHASE_CONDITIONING_TRACE.md → Section 1
- PHASE_CONDITIONING_SUMMARY.md → "Stage 1: Command-Line Argument"
- PHASE_CONDITIONING_QUICK_REF.md → Line 1 of file table

**Phase Extraction:**
- PHASE_CONDITIONING_TRACE.md → Sections 3, 4
- PHASE_CONDITIONING_SUMMARY.md → "Stage 2" + "Phase Extraction Method"
- PHASE_CONDITIONING_VISUAL_GUIDE.txt → GEN_TRAIN_DATALOADER section
- README_PHASE_CONDITIONING.md → "Phase Extraction Method"

**Data Loading:**
- PHASE_CONDITIONING_TRACE.md → Section 3, 5, 6
- PHASE_CONDITIONING_SUMMARY.md → Stages 2-4
- PHASE_CONDITIONING_VISUAL_GUIDE.txt → Flow diagram

**Model Architecture:**
- PHASE_CONDITIONING_TRACE.md → Sections 9, 10
- PHASE_CONDITIONING_SUMMARY.md → Stages 5 + "Key Implementation Details"
- PHASE_CONDITIONING_VISUAL_GUIDE.txt → Architecture comparison
- README_PHASE_CONDITIONING.md → "Architecture Changes"

**Training Loop:**
- PHASE_CONDITIONING_TRACE.md → Sections 6, 7
- PHASE_CONDITIONING_SUMMARY.md → Stage 4
- PHASE_CONDITIONING_VISUAL_GUIDE.txt → TRAINING LOOP section

**Forward Pass:**
- PHASE_CONDITIONING_TRACE.md → Sections 7, 10
- PHASE_CONDITIONING_SUMMARY.md → Stage 5
- PHASE_CONDITIONING_VISUAL_GUIDE.txt → MODEL FORWARD section

**Issues & Fixes:**
- PHASE_CONDITIONING_SUMMARY.md → "Discovered Issues"
- README_PHASE_CONDITIONING.md → "Known Issues"
- PHASE_CONDITIONING_QUICK_REF.md → "Common Issues & Debugging"

---

## Key Facts Summary

| Aspect | Detail |
|--------|--------|
| **Flag Name** | `--use_phase_conditioning` |
| **Type** | Boolean flag (action="store_true") |
| **Default** | False (disabled) |
| **Definition File** | src/utils/util.py:215-219 |
| **Phase Source** | zarr["position"] data |
| **Phase Shape** | (T, H) - per-frame, per-row |
| **Extraction Method** | Hilbert transform |
| **Batch Format** | 7-tuple (with) vs 5-tuple (without) |
| **U-Net Channels** | 3*(T-1) (with) vs T-1 (without) |
| **BS-Net L0 Channels** | 3 (with) vs 1 (without) |
| **BS-Net Deeper** | 1x1 projections (with) vs scalar mult (without) |
| **Rotations** | Disabled (with) vs enabled (without) |
| **Flips** | Applied to phase too (with) vs image only (without) |
| **Status** | Complete (except train.py has 2 missing params) |

---

## Using These Documents

### For Code Review:
1. PHASE_CONDITIONING_QUICK_REF.md (identify file/location)
2. PHASE_CONDITIONING_TRACE.md (get context and full implementation)

### For Bug Fixing:
1. PHASE_CONDITIONING_QUICK_REF.md ("Common Issues & Debugging")
2. PHASE_CONDITIONING_SUMMARY.md ("Discovered Issues")
3. README_PHASE_CONDITIONING.md ("Known Issues")
4. PHASE_CONDITIONING_TRACE.md (if still unsure)

### For Feature Development:
1. PHASE_CONDITIONING_SUMMARY.md (understand architecture)
2. PHASE_CONDITIONING_TRACE.md (find where to add code)
3. PHASE_CONDITIONING_VISUAL_GUIDE.txt (visualize impact)

### For Training:
1. README_PHASE_CONDITIONING.md ("Quick Start" section)
2. PHASE_CONDITIONING_SUMMARY.md ("Critical Code Paths")
3. PHASE_CONDITIONING_QUICK_REF.md (verify tensor shapes)

### For Testing:
1. README_PHASE_CONDITIONING.md ("Testing" section)
2. PHASE_CONDITIONING_QUICK_REF.md ("Testing Phase Conditioning")
3. PHASE_CONDITIONING_TRACE.md ("Distributed Training Special Handling")

---

## Document Statistics

| Document | Size | Sections | Purpose |
|----------|------|----------|---------|
| README_PHASE_CONDITIONING.md | 7.3 KB | 12 | Navigation & overview |
| PHASE_CONDITIONING_VISUAL_GUIDE.txt | 5.8 KB | 5 | Diagrams & visuals |
| PHASE_CONDITIONING_SUMMARY.md | 11 KB | 12 | Executive summary |
| PHASE_CONDITIONING_QUICK_REF.md | 6.0 KB | 8 | Quick lookup |
| PHASE_CONDITIONING_TRACE.md | 24 KB | 15 | Detailed reference |
| **TOTAL** | **54 KB** | **52** | Complete documentation |

---

## Quick Problem Solver

**Problem: Phase not being extracted**
→ Check: PHASE_CONDITIONING_QUICK_REF.md Issue #1

**Problem: Shape mismatch in model**
→ Check: PHASE_CONDITIONING_QUICK_REF.md Issue #2

**Problem: DDP errors about unused parameters**
→ Check: PHASE_CONDITIONING_QUICK_REF.md Issue #3

**Problem: Want to understand phase conditioning**
→ Read: PHASE_CONDITIONING_SUMMARY.md (20 min) + VISUAL_GUIDE (5 min)

**Problem: Want to modify code**
→ Use: PHASE_CONDITIONING_QUICK_REF.md (find location) + TRACE.md (understand)

**Problem: Want all details**
→ Read: PHASE_CONDITIONING_TRACE.md (comprehensive)

---

## Integration with Codebase

These documents map directly to:
- 5 main source files
- 70+ code locations
- 15 major sections
- Complete pipeline from CLI to model output

All file/line references are current as of the analysis date.

---

## Maintenance Notes

- **Last Updated:** April 27, 2026
- **Scope:** Complete --use_phase_conditioning flag implementation
- **Status:** All code paths documented
- **Known Gaps:** train.py missing 2 parameters (documented in issues)
- **Test Coverage:** 8 test files reference phase conditioning

---

## How to Use This Index

1. **Find what you need:** Look at "Quick Navigation" at top
2. **Go to recommended document:** Read the suggested file
3. **Need more detail?** Cross-reference using "File Cross-Reference" section
4. **Stuck?** Check "Quick Problem Solver"
5. **Want everything?** Read documents in order listed in navigation

---

**Recommended Starting Points:**
- Beginner: README_PHASE_CONDITIONING.md
- Visual learner: PHASE_CONDITIONING_VISUAL_GUIDE.txt  
- Practical: PHASE_CONDITIONING_QUICK_REF.md
- Thorough: PHASE_CONDITIONING_SUMMARY.md
- Comprehensive: PHASE_CONDITIONING_TRACE.md

