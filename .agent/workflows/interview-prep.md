---
description: Create interview and exam preparation content for a concept or project
---

# Interview & Exam Preparation Workflow

This workflow creates comprehensive interview questions and exam preparation content.

## ⚠️ IMPORTANT RULES

### Rule 1: Explain Like Teaching a 10-Year-Old
- Every technical term MUST be explained in simple words
- Use real-life analogies (games, school, toys, food, sports)
- NO unexplained jargon allowed

### Rule 2: Use Mermaid Diagrams
- Every concept should have at least one Mermaid diagram
- Diagrams make understanding 10x easier

---

## Part 1: Interview Questions File (`interview_questions.md`)

### Requirements:
- **MINIMUM 10-20 Questions**
- Cover ALL difficulty levels (Easy → Medium → Hard)

### Question Categories:
1. **Basic Concept Questions** (What is...?) - 3-5 questions
2. **Why/When Questions** (Why do we use...?) - 3-5 questions
3. **How Questions** (How does... work?) - 3-5 questions
4. **Comparison Questions** (Difference between A and B?) - 2-3 questions
5. **Scenario Questions** (What would happen if...?) - 2-3 questions
6. **Troubleshooting Questions** (How to fix...?) - 2-3 questions

### Format for EACH Question:

```markdown
---

## Q[Number]: [Question Title]

### 🎯 The Question
[Exact interview question as interviewer would ask]

### 💡 Simple Answer (Like Explaining to a Child)
[Use simple words, real-life analogy, NO jargon]

**Analogy:** [Real-life comparison]

### 📚 Technical Answer (For Interviewer)
[Detailed technical explanation with proper terms]

### 🎨 Visual Explanation
```mermaid
[Diagram showing the concept]
```

### 🌍 Real-Life Example
[Practical application or scenario]

### ⚠️ Common Mistakes (What NOT to Say)
- ❌ Don't say: [Wrong answer]
- ❌ Don't say: [Wrong answer]

### ✅ Key Points to Remember
- ✓ Point 1
- ✓ Point 2
- ✓ Point 3

---
```

---

## Part 2: Exam Preparation File (`exam_preparation.md`)

### Section A: Multiple Choice Questions (MCQ)
**Minimum: 10 Questions**

```markdown
## MCQ [Number]

**Question:** [Question text]

**Options:**
- A) [Option 1]
- B) [Option 2]
- C) [Option 3]
- D) [Option 4]

**✅ Correct Answer:** [Letter]

**📖 Simple Explanation:** [Why correct - in simple words]

**❌ Why Others Are Wrong:**
- A) [Reason - simple language]
- B) [Reason - simple language]
- C) [Reason - simple language]
- D) [Reason - simple language]

---
```

### Section B: Multiple Select Questions (MSQ)
**Minimum: 5 Questions**

```markdown
## MSQ [Number]

**Question:** [Question text] *(Select ALL that apply)*

**Options:**
- A) [Option 1]
- B) [Option 2]
- C) [Option 3]
- D) [Option 4]
- E) [Option 5]

**✅ Correct Answers:** [Letters, e.g., A, C, D]

**📖 Explanation:**
- A) ✓ Correct because: [reason]
- C) ✓ Correct because: [reason]
- D) ✓ Correct because: [reason]
- B) ✗ Wrong because: [reason]
- E) ✗ Wrong because: [reason]

---
```

### Section C: Numerical/Calculation Questions
**Minimum: 5 Questions**

```markdown
## Numerical [Number]

**Question:** [Question with numbers]

**Given:**
- Value 1 = X
- Value 2 = Y
- Value 3 = Z

**🧮 Solution Steps:**

**Step 1:** [What we do first]
```
[Calculation]
```

**Step 2:** [What we do next]
```
[Calculation]
```

**Step 3:** [Final step]
```
[Calculation]
```

**✅ Final Answer:** [Number with units]

**📖 Simple Explanation:** [Why we did these steps - like explaining to a child]

---
```

### Section D: Fill in the Blanks
**Minimum: 5 Questions**

```markdown
## Fill [Number]

**Question:** The process of _______ is used when we want to _______.

**✅ Answers:** [word 1], [word 2]

**📖 Explanation:** [Simple explanation of the complete sentence]

---
```

### Section E: True/False (BONUS)
**Minimum: 5 Questions**

```markdown
## T/F [Number]

**Statement:** [Statement that is either true or false]

**✅ Answer:** True / False

**📖 Explanation:** [Why it's true/false - simple language]

---
```

---

## Part 3: Quick Revision Sheet (`interview_preparation.md`)

### Content:
1. **30-Second Summary** - Entire topic in 5-6 bullets
2. **Jargon Glossary** - Technical word → Simple meaning table
3. **Top 10 Things to Remember** - Numbered list
4. **Comparison Tables** - Side-by-side comparisons
5. **Cheat Sheet** - Copy-paste ready formulas/syntax
6. **Common Traps** - What interviewers try to trick you with
7. **Mermaid Summary** - One diagram showing everything
