---
description: Explain function/method with argument-by-argument breakdown (3.1-3.7)
---

# Function/Method Argument Explanation Workflow

This workflow explains every parameter/argument of a function or method following the strict 3.1-3.7 format.

## Prerequisites
- The function/method call to explain
- Context of usage

---

## Explanation Format (MANDATORY)

For EVERY function or method call, create explanation with ALL these sections for EACH parameter:

```markdown
### ⚙️ Function/Method: `function_name()`

**Full Call:**
```python
function_name(param1=value1, param2=value2, ...)
```

---

### Parameter: `param1`

#### 3.1 What It Does
- Purpose of this parameter
- What it controls/specifies

#### 3.2 Why It Is Used
- Problem it solves
- **Is this the only way?**
  - Alternative parameters
  - Why this parameter is needed

#### 3.3 When to Use It
- Scenarios requiring this parameter
- Conditions for different values

#### 3.4 Where to Use It
- Real-world applications
- Project types

#### 3.5 How to Use It (Syntax + Example)
- **Syntax:**
```python
function_name(param1=value)
```
- **Example:** [code example]

#### 3.6 How It Affects Execution Internally
- Internal behavior change
- Processing impact
- Memory/performance effects

#### 3.7 Output Impact with Examples
- **With default value:**
```python
[code with default]
```
Output: [output]

- **With custom value:**
```python
[code with custom]
```
Output: [output]

---

### Parameter: `param2`
[Repeat 3.1-3.7]

---

### Default Values Explanation

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `param1` | [default] | [what default does] |
| `param2` | [default] | [what default does] |
```

---

## Example: Explaining `train_test_split()`

### ⚙️ Function: `train_test_split()`

**Full Call:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)
```

---

### Parameter: `test_size`

#### 3.1 What It Does
Specifies the proportion of data to reserve for testing.

#### 3.2 Why It Is Used
We need separate data to evaluate model performance fairly.

**Alternatives:**
| Approach | Pros | Cons |
|----------|------|------|
| `test_size=0.2` | Common standard | May not suit all cases |
| `train_size=0.8` | Equivalent | Less intuitive |

#### 3.3 When to Use It
- Always when splitting data for ML
- Adjust based on dataset size

#### 3.4 Where to Use It
- Every ML project
- Model validation
- Cross-validation setup

#### 3.5 How to Use It
```python
# 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

#### 3.6 How It Affects Execution Internally
1. Calculates number of samples: `n_test = int(n_samples * test_size)`
2. Shuffles data (if shuffle=True)
3. Splits at calculated index
4. Returns four arrays

#### 3.7 Output Impact with Examples
**With `test_size=0.2`:**
```python
X = [[1], [2], [3], [4], [5]]
X_train, X_test = train_test_split(X, test_size=0.2)
print(len(X_train), len(X_test))
```
Output: `4 1`

**With `test_size=0.5`:**
Output: `2 3` (or `3 2`)

---

### Parameter: `random_state`

#### 3.1 What It Does
Sets the seed for random number generator to ensure reproducible splits.

#### 3.2 Why It Is Used
Without it, every run produces different splits, making debugging impossible.

**Analogy:** Like saving your game - you can always restart from the same point.

#### 3.3 When to Use It
- ALWAYS in development/debugging
- When results must be reproducible
- In tutorials and examples

#### 3.4 Where to Use It
- All ML experiments
- Teaching materials
- Reproducible research

#### 3.5 How to Use It
```python
# Reproducible split
train_test_split(X, y, random_state=42)

# Different each time
train_test_split(X, y)  # No random_state
```

#### 3.6 How It Affects Execution Internally
1. Initializes numpy RandomState with seed
2. Uses this state for shuffle operation
3. Same seed = same shuffle = same split

#### 3.7 Output Impact
**With `random_state=42`:**
Run 1: `[3, 1, 4, 2, 5]` → Always same
Run 2: `[3, 1, 4, 2, 5]` → Always same

**Without `random_state`:**
Run 1: `[2, 5, 1, 3, 4]` → Random
Run 2: `[4, 1, 3, 5, 2]` → Different

---

### Default Values Summary

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `test_size` | 0.25 | 25% for testing |
| `train_size` | None | Complement of test_size |
| `random_state` | None | Random each run |
| `shuffle` | True | Shuffle before split |
| `stratify` | None | No stratification |

---

## Validation Checklist
- [ ] Every parameter has 3.1-3.7 sections
- [ ] Default values explained
- [ ] Comparison table for alternatives
- [ ] Multiple examples provided
- [ ] Internal behavior explained
- [ ] Output impact demonstrated
