---
description: Create the 7 mandatory documentation files (Section 11) with strict content requirements.
---

# Create Documentation Files Workflow

This workflow generates the **7 Mandatory Documentation Files**.
**Location:** `<project_name>/documentation/`

## Important Rule: Explain Like Teaching a 10-Year-Old
- **NO unexplained jargon** - Every technical term MUST be explained in simple words
- Use **real-life analogies** (games, school, toys, food)
- Use **Mermaid diagrams** wherever possible for visual understanding

---

## Execution Order (STRICT)

### 1. `Original_Problem.md` (FIRST - MANDATORY)
**Purpose:** Preserve the EXACT user input.
**Content:** 
- Copy-paste the EXACT problem statement/usecase provided by the user
- NO modifications, NO additional explanations
- Just the raw problem as given

---

### 2. `problem_statement.md`
**Purpose:** The WHAT and WHY (simplified).
**Required Sections:**
1.  **Problem Statement**: Define problem like explaining to a child
2.  **Real-Life Analogy**: Compare to everyday situations
3.  **Steps to Solve**: Simple step-by-step breakdown
4.  **Expected Output**: What we expect (with examples)
5.  **Mermaid Diagram**: Flow of the problem → solution

---

### 3. `concepts_explained.md` (CORE THEORY)
**Purpose:** Deep dive into every concept and import.
**Important:** Every jargon word MUST have a "Simple Explanation" box.

**Structure:** For **EACH** concept/import, include these **12 Points**:
1.  **Definition**: Simple, 10-year-old friendly
2.  **Simple Analogy**: Like what in real life?
3.  **Why it is used**: What problem it solves
4.  **When to use it**: Best conditions
5.  **Where to use it**: Real-world examples
6.  **Is this the only way?**: Alternatives comparison table
7.  **Mermaid Diagram**: Visual explanation
8.  **How to use it**: Syntax + Simple code example
9.  **How it works internally**: Step-by-step (like recipe steps)
10. **Visual Summary**: Bullet/Flow recap
11. **Advantages & Disadvantages**: Pros/Cons table
12. **Jargon Glossary**: All technical terms explained simply

---

### 4. `observations_and_conclusion.md`
**Purpose:** Interpreting the results.
**Required Sections:**
1.  **Execution Output**: Actual raw output
2.  **Output Explanation**: What does each number/result mean?
3.  **Mermaid Diagram**: Visual flow of results
4.  **Observations**: Patterns noticed (simple language)
5.  **Insights**: What actions to take based on results
6.  **Conclusion**: Summary in simple words

---

### 5. `interview_questions.md` (NEW - SEPARATE FILE)
**Purpose:** 10-20 Interview Questions with thorough answers.
**Format for EACH Question:**

```markdown
## Q1: [Question Title]

### 🎯 The Question
[Exact interview question]

### 💡 Simple Answer (For 10-Year-Old)
[Explain like teaching a child]

### 📚 Technical Answer (For Interviewer)
[Detailed technical explanation]

### 🎨 Mermaid Diagram
[Visual explanation using Mermaid]

### 🌍 Real-Life Analogy
[Compare to everyday situation]

### ⚠️ Common Mistakes
[What NOT to say]

### ✅ Key Points to Remember
- Point 1
- Point 2
- Point 3
```

**Minimum:** 10-20 questions covering:
- Basic concept questions (What is...?)
- Why/When questions (Why do we use...?)
- How questions (How does... work?)
- Comparison questions (Difference between A and B?)
- Scenario questions (What would happen if...?)
- Troubleshooting questions (How to fix...?)

---

### 6. `exam_preparation.md` (NEW - SEPARATE FILE)
**Purpose:** Practice questions for exams.
**Required Question Types:**

#### Section A: Multiple Choice Questions (MCQ) - 10+ Questions
```markdown
## MCQ 1
**Question:** [Question text]

**Options:**
- A) Option 1
- B) Option 2
- C) Option 3
- D) Option 4

**✅ Correct Answer:** [Letter]

**📖 Explanation:** [Why this is correct - simple language]

**❌ Why Others Are Wrong:**
- A) [Why wrong]
- B) [Why wrong]
...
```

#### Section B: Multiple Select Questions (MSQ) - 5+ Questions
```markdown
## MSQ 1
**Question:** [Question text] (Select ALL that apply)

**Options:**
- A) Option 1
- B) Option 2
- C) Option 3
- D) Option 4
- E) Option 5

**✅ Correct Answers:** [Letters, e.g., A, C, D]

**📖 Explanation:** [Why each correct option is correct]
```

#### Section C: Numerical/Calculation Questions - 5+ Questions
```markdown
## Numerical 1
**Question:** [Question with numbers]

**Given:**
- Value 1 = X
- Value 2 = Y

**Solution Steps:**
1. Step 1: [Calculation]
2. Step 2: [Calculation]
3. Step 3: [Final answer]

**✅ Final Answer:** [Number with units]

**🎨 Mermaid Diagram:** [If helpful for understanding]
```

#### Section D: Fill in the Blanks - 5+ Questions
```markdown
## Fill 1
**Question:** The process of _______ is used to...

**Answer:** [Correct word/phrase]

**Explanation:** [Simple explanation]
```

---

### 7. `interview_preparation.md` (QUICK REVISION)
**Purpose:** One-page quick revision sheet.
**Required Sections:**
1.  **30-Second Summary**: Entire topic in 5-6 bullets
2.  **Key Terms Glossary**: Technical word → Simple meaning
3.  **Most Important Points**: Top 10 things to remember
4.  **Quick Comparison Tables**: Side-by-side comparisons
5.  **Formula/Syntax Cheat Sheet**: Copy-paste ready
6.  **Common Interview Traps**: What NOT to say
7.  **Mermaid Summary Diagram**: Visual overview
