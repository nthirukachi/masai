# AS24: Multiple Linear Regression - Exam Preparation

> 📚 **Purpose:** Quick revision, MCQs, MSQs, Numerical Problems, Fill-in-the-blanks, Shortcuts
> 📘 **Classroom Files:** [Part 1](./AS24_MultipleLinearRegression1.md) | [Part 2](./AS24_MultipleLinearRegression2.md) | [Part 3](./AS24_MultipleLinearRegression3.md) | [Part 4](./AS24_MultipleLinearRegression4.md)

---

## 📋 Quick Revision Points

### Core Concepts - One-Liner Summaries

| # | Concept | One-Line Summary |
|---|---------|------------------|
| 1 | **MLR Definition** | Predicting Y using 2+ independent variables: Y = β₀ + β₁X₁ + ... + βₙXₙ + ε |
| 2 | **Simple vs MLR** | Simple has 1 predictor, MLR has 2+ predictors |
| 3 | **Hyperplane** | A plane in n-dimensions; generalizes line (2D) and plane (3D) |
| 4 | **β Coefficient** | Change in Y for 1-unit change in X, holding other Xs constant |
| 5 | **Intercept (β₀)** | Predicted Y when all X variables are zero |
| 6 | **OLS** | Ordinary Least Squares - minimizes sum of squared residuals |
| 7 | **Linearity Assumption** | Relationship between each X and Y must be linear |
| 8 | **Independence of Errors** | Residuals should not be correlated with each other |
| 9 | **Homoscedasticity** | Variance of residuals should be constant across all X values |
| 10 | **Normality of Residuals** | Residuals should follow normal distribution |
| 11 | **Multicollinearity** | When predictor variables are highly correlated with each other |
| 12 | **VIF** | Variance Inflation Factor; VIF > 5 concerning, > 10 serious |
| 13 | **VIF Formula** | VIF = 1 / (1 - R²ᵢ) where R²ᵢ is from regressing Xᵢ on other Xs |
| 14 | **One-Hot Encoding** | Create k-1 dummy columns for k categories |
| 15 | **Label Encoding** | Assign numbers to categories (only for ordinal data) |
| 16 | **Dummy Variable Trap** | Perfect multicollinearity if all k dummies included |
| 17 | **R²** | Proportion of variance in Y explained by Xs (0 to 1) |
| 18 | **Adjusted R²** | R² with penalty for unnecessary variables |
| 19 | **R² Formula** | R² = 1 - (SS_res / SS_tot) |
| 20 | **Adj R² Formula** | R²_adj = 1 - (1-R²)(n-1)/(n-p-1) |

---

## 📝 MCQs (Single Correct Answer)

### MCQ 1
**In Multiple Linear Regression, what does the coefficient β₁ represent?**
- A) The intercept of the regression line
- B) The change in Y for a one-unit change in X₁, holding other variables constant
- C) The correlation between X₁ and Y
- D) The total variance explained by X₁

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** The coefficient β₁ represents the change in the dependent variable Y for each one-unit increase in X₁, while keeping all other predictor variables constant (ceteris paribus).
</details>

---

### MCQ 2
**What is the primary difference between Simple Linear Regression and Multiple Linear Regression?**
- A) The type of dependent variable
- B) The number of independent variables
- C) The error term calculation
- D) The method of finding coefficients

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** Simple LR has 1 independent variable, while MLR has 2 or more independent variables. Both use OLS and have error terms.
</details>

---

### MCQ 3
**Which of the following VIF values indicates serious multicollinearity?**
- A) VIF = 1.0
- B) VIF = 2.5
- C) VIF = 4.8
- D) VIF = 12.5

<details>
<summary>Show Answer</summary>

**Answer: D**

**Explanation:** VIF = 1 indicates no multicollinearity. VIF 1-5 is acceptable. VIF > 5 is concerning, VIF > 10 is serious. VIF = 12.5 requires action.
</details>

---

### MCQ 4
**If R² = 0.8 and you add a useless random variable, what happens to R² and Adjusted R²?**
- A) Both increase
- B) R² increases, Adjusted R² decreases
- C) Both decrease
- D) R² stays same, Adjusted R² decreases

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** R² ALWAYS increases (or stays same) with more variables. Adjusted R² penalizes useless variables and will DECREASE if the variable doesn't improve the model.
</details>

---

### MCQ 5
**When should you use One-Hot Encoding instead of Label Encoding?**
- A) When the variable is ordinal (has natural order)
- B) When the variable is nominal (no natural order)
- C) When there are only 2 categories
- D) When the variable is continuous

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** One-Hot Encoding is preferred for nominal categories (no order) like city names. Label Encoding is only appropriate for ordinal categories (natural order) like education levels.
</details>

---

### MCQ 6
**The Dummy Variable Trap occurs when:**
- A) Categorical variable has too few categories
- B) All k dummy variables are included for k categories
- C) Label encoding is used instead of one-hot
- D) VIF is too high

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** Including all k dummies creates perfect multicollinearity because their sum always equals 1. Solution: Include only k-1 dummies.
</details>

---

### MCQ 7
**What does a negative coefficient for "Number of Rooms" in a house price prediction model most likely indicate?**
- A) More rooms actually decrease price
- B) Multicollinearity with another correlated variable
- C) The model is overfitting
- D) The data has outliers

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** Logically, more rooms should increase price. A negative coefficient usually indicates multicollinearity, where another correlated variable (like house size) is capturing the effect.
</details>

---

### MCQ 8
**For a dataset with n=100 observations and p=5 predictors, if R²=0.75, what is the Adjusted R²?**
- A) 0.75
- B) 0.737
- C) 0.763
- D) 0.720

<details>
<summary>Show Answer</summary>

**Answer: B**

**Calculation:**
```
Adj R² = 1 - (1 - 0.75)(100-1)/(100-5-1)
       = 1 - (0.25)(99)/(94)
       = 1 - 24.75/94
       = 1 - 0.263
       = 0.737
```
</details>

---

### MCQ 9
**Which assumption is UNIQUE to Multiple Linear Regression (not present in Simple LR)?**
- A) Linearity
- B) Homoscedasticity
- C) Normality of residuals
- D) No multicollinearity

<details>
<summary>Show Answer</summary>

**Answer: D**

**Explanation:** Multicollinearity only exists when there are 2+ predictors. Simple LR has only 1 predictor, so this assumption cannot apply.
</details>

---

### MCQ 10
**What does VIF = 1 indicate?**
- A) Severe multicollinearity
- B) Moderate multicollinearity
- C) No multicollinearity
- D) The variable should be removed

<details>
<summary>Show Answer</summary>

**Answer: C**

**Explanation:** VIF = 1 means R²ᵢ = 0 (the variable cannot be predicted by other Xs). This indicates NO multicollinearity - perfect!
</details>

---

### MCQ 11
**In the equation Y = 5 + 2X₁ - 3X₂, if X₁ increases by 1 unit while X₂ is held constant, Y will:**
- A) Increase by 5
- B) Increase by 2
- C) Decrease by 3
- D) Increase by 4

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** The coefficient of X₁ is +2. So for each 1-unit increase in X₁ (holding X₂ constant), Y increases by 2.
</details>

---

### MCQ 12
**What is the purpose of a Q-Q plot in regression analysis?**
- A) Check for linearity
- B) Check for multicollinearity
- C) Check for normality of residuals
- D) Check for homoscedasticity

<details>
<summary>Show Answer</summary>

**Answer: C**

**Explanation:** Q-Q (Quantile-Quantile) plot compares residual distribution to normal distribution. Points on diagonal = normal residuals.
</details>

---

### MCQ 13
**Homoscedasticity is violated when:**
- A) Residuals have constant variance
- B) Residuals increase as predicted values increase
- C) Residuals are normally distributed
- D) Residuals are independent

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** Homoscedasticity means CONSTANT variance. If variance changes with predicted values (like a funnel shape), it's heteroscedasticity - a violation.
</details>

---

### MCQ 14
**Which method is used to find optimal coefficients in standard MLR?**
- A) Gradient Descent
- B) Maximum Likelihood
- C) Ordinary Least Squares
- D) Newton-Raphson

<details>
<summary>Show Answer</summary>

**Answer: C**

**Explanation:** OLS (Ordinary Least Squares) minimizes the sum of squared residuals to find optimal β values.
</details>

---

### MCQ 15
**If you have a categorical variable "City" with 5 categories, how many dummy columns should you create?**
- A) 5
- B) 4
- C) 6
- D) 3

<details>
<summary>Show Answer</summary>

**Answer: B**

**Explanation:** For k categories, create k-1 dummies to avoid dummy variable trap. 5 categories → 4 dummy columns.
</details>

---

## 📝 MSQs (Multiple Correct Answers)

### MSQ 1
**Which of the following are valid methods to detect multicollinearity? (Select all that apply)**
- [ ] A) Correlation matrix
- [ ] B) VIF
- [ ] C) Q-Q plot
- [ ] D) Pair plots
- [ ] E) R² score

<details>
<summary>Show Answer</summary>

**Answers: A, B, D**

**Explanation:**
- A) Correlation matrix shows pairwise correlations ✓
- B) VIF quantifies multicollinearity ✓
- C) Q-Q plot checks normality, not multicollinearity ✗
- D) Pair plots visualize relationships between variables ✓
- E) R² measures model fit, not multicollinearity ✗
</details>

---

### MSQ 2
**Which are valid solutions for multicollinearity? (Select all that apply)**
- [ ] A) Remove one of the correlated variables
- [ ] B) Use PCA
- [ ] C) Increase sample size
- [ ] D) Use Ridge regression
- [ ] E) Combine correlated variables

<details>
<summary>Show Answer</summary>

**Answers: A, B, D, E**

**Explanation:**
- A) Removing one variable eliminates redundancy ✓
- B) PCA creates uncorrelated components ✓
- C) Increasing sample size doesn't fix correlation ✗
- D) Ridge regression shrinks correlated coefficients ✓
- E) Combining creates a single variable ✓
</details>

---

### MSQ 3
**Which statements about R² are TRUE? (Select all that apply)**
- [ ] A) R² always increases when adding variables
- [ ] B) R² ranges from -∞ to 1
- [ ] C) R² = 1 means perfect prediction on training data
- [ ] D) R² can never be negative
- [ ] E) Higher R² always means better model

<details>
<summary>Show Answer</summary>

**Answers: A, B, C**

**Explanation:**
- A) True - R² never decreases with more variables ✓
- B) True - R² can be negative for very bad models ✓
- C) True - R² = 1 means all variance explained ✓
- D) False - R² can be negative if model is worse than mean ✗
- E) False - High R² might indicate overfitting ✗
</details>

---

### MSQ 4
**Which are assumptions of Multiple Linear Regression? (Select all that apply)**
- [ ] A) Linear relationship between Y and each X
- [ ] B) Normally distributed X variables
- [ ] C) Constant variance of residuals
- [ ] D) No multicollinearity among predictors
- [ ] E) Independent residuals

<details>
<summary>Show Answer</summary>

**Answers: A, C, D, E**

**Explanation:**
- A) Linearity assumption ✓
- B) X doesn't need to be normal; residuals do ✗
- C) Homoscedasticity ✓
- D) No multicollinearity ✓
- E) Independence of errors ✓
</details>

---

### MSQ 5
**When comparing Model A (3 features, Adj R² = 0.72) and Model B (5 features, Adj R² = 0.71), which statements are TRUE?**
- [ ] A) Model B has higher R²
- [ ] B) Model A is preferred based on Adjusted R²
- [ ] C) Model B's additional features are not adding value
- [ ] D) Model A will definitely generalize better
- [ ] E) Model B has more overfitting risk

<details>
<summary>Show Answer</summary>

**Answers: A, B, C, E**

**Explanation:**
- A) More features → higher R² (always) ✓
- B) Higher Adj R² = Model A is better ✓
- C) Adj R² dropped, so new features aren't useful ✓
- D) "Definitely" is too strong; we can't be 100% sure ✗
- E) More features = more overfitting potential ✓
</details>

---

### MSQ 6
**Which techniques can be used to check if linearity assumption is satisfied? (Select all that apply)**
- [ ] A) Scatter plots of X vs Y
- [ ] B) Residual vs Fitted plot
- [ ] C) Correlation matrix
- [ ] D) Component-component plus residual plot
- [ ] E) VIF calculation

<details>
<summary>Show Answer</summary>

**Answers: A, B, D**

**Explanation:**
- A) Scatter plots show linear/non-linear patterns ✓
- B) Non-random patterns indicate non-linearity ✓
- C) Correlation measures strength, not linearity ✗
- D) CCPR plots help detect non-linearity ✓
- E) VIF checks multicollinearity, not linearity ✗
</details>

---

### MSQ 7
**Which are valid interpretations of VIF = 10 for variable X₁? (Select all that apply)**
- [ ] A) 90% of X₁'s variance is explained by other Xs
- [ ] B) The standard error of β₁ is inflated by factor of √10
- [ ] C) X₁ should definitely be removed
- [ ] D) X₁ has high correlation with at least one other variable
- [ ] E) The coefficient β₁ is unreliable

<details>
<summary>Show Answer</summary>

**Answers: A, B, D, E**

**Explanation:**
- A) VIF=10 → R²=0.9 → 90% explained ✓
- B) SE inflation = √VIF = √10 ≈ 3.16 ✓
- C) Should be investigated, not "definitely" removed ✗
- D) High VIF indicates correlation with other Xs ✓
- E) High VIF makes coefficient estimates unstable ✓
</details>

---

### MSQ 8
**Which transformations might help if linearity assumption is violated? (Select all that apply)**
- [ ] A) Log transformation of Y
- [ ] B) Square root of X
- [ ] C) Adding polynomial terms (X²)
- [ ] D) Standardization (Z-score)
- [ ] E) Removing outliers

<details>
<summary>Show Answer</summary>

**Answers: A, B, C**

**Explanation:**
- A) Log Y can linearize exponential relationships ✓
- B) sqrt(X) can linearize square-root relationships ✓
- C) Adding X² captures curvature ✓
- D) Standardization doesn't change linearity ✗
- E) Outlier removal doesn't fix non-linearity ✗
</details>

---

### MSQ 9
**For One-Hot Encoding with drop_first=True, which are TRUE? (Select all that apply)**
- [ ] A) The dropped category becomes the baseline/reference
- [ ] B) Other coefficients are compared to this baseline
- [ ] C) It prevents dummy variable trap
- [ ] D) Which category to drop doesn't matter mathematically
- [ ] E) You always drop the most frequent category

<details>
<summary>Show Answer</summary>

**Answers: A, B, C, D**

**Explanation:**
- A) Dropped category is the reference ✓
- B) Other β's represent difference from reference ✓
- C) Prevents perfect multicollinearity ✓
- D) Any category can be dropped; results are equivalent ✓
- E) Dropping strategy doesn't matter mathematically ✗
</details>

---

### MSQ 10
**Which are potential consequences of ignoring multicollinearity? (Select all that apply)**
- [ ] A) Unstable coefficient estimates
- [ ] B) Sign flips in coefficients
- [ ] C) Lower R² value
- [ ] D) Wide confidence intervals
- [ ] E) Difficulty interpreting individual effects

<details>
<summary>Show Answer</summary>

**Answers: A, B, D, E**

**Explanation:**
- A) Small data changes → big coefficient changes ✓
- B) Positive becomes negative unexpectedly ✓
- C) R² may actually be artificially high ✗
- D) Large standard errors → wide CIs ✓
- E) Can't isolate individual variable effects ✓
</details>

---

## 🔢 Numerical Questions

### Numerical 1
**Given the regression equation: Price = 10 + 5×Size + 3×Rooms - 0.5×Age**

**Calculate the predicted price for:**
- Size = 2000 sq ft (in units of 1000)
- Rooms = 4
- Age = 10 years

<details>
<summary>Show Answer</summary>

**Solution:**
```
Price = 10 + 5(2) + 3(4) - 0.5(10)
      = 10 + 10 + 12 - 5
      = 27 (units, e.g., ₹27 Lakhs)
```
</details>

---

### Numerical 2
**Calculate VIF if R² = 0.85 when regressing X₁ on all other X variables.**

<details>
<summary>Show Answer</summary>

**Solution:**
```
VIF = 1 / (1 - R²)
    = 1 / (1 - 0.85)
    = 1 / 0.15
    = 6.67

Interpretation: VIF > 5, concerning but not severe
```
</details>

---

### Numerical 3
**Calculate Adjusted R² given:**
- R² = 0.80
- n = 50 observations
- p = 4 predictors

<details>
<summary>Show Answer</summary>

**Solution:**
```
Adj R² = 1 - (1 - R²)(n - 1) / (n - p - 1)
       = 1 - (1 - 0.80)(50 - 1) / (50 - 4 - 1)
       = 1 - (0.20)(49) / (45)
       = 1 - 9.8 / 45
       = 1 - 0.2178
       = 0.7822
```
</details>

---

### Numerical 4
**Model A: R² = 0.75, p = 3**
**Model B: R² = 0.77, p = 5**
**n = 100 for both**

**Which model has higher Adjusted R²? Show calculations.**

<details>
<summary>Show Answer</summary>

**Solution:**
```
Model A:
Adj R² = 1 - (0.25)(99)/(100-3-1) = 1 - 24.75/96 = 1 - 0.258 = 0.742

Model B:
Adj R² = 1 - (0.23)(99)/(100-5-1) = 1 - 22.77/94 = 1 - 0.242 = 0.758

Model B has higher Adjusted R² (0.758 > 0.742)
Despite adding 2 more variables, Model B still performs better!
The extra variables ARE adding value.
```
</details>

---

### Numerical 5
**If SSE (Sum of Squared Errors) = 100 and SST (Total Sum of Squares) = 500, calculate R².**

<details>
<summary>Show Answer</summary>

**Solution:**
```
R² = 1 - SSE/SST
   = 1 - 100/500
   = 1 - 0.20
   = 0.80

Model explains 80% of variance!
```
</details>

---

### Numerical 6
**For a categorical variable "Size" with categories {Small, Medium, Large, XL}, how many dummy variables are created using One-Hot Encoding with drop_first=True?**

<details>
<summary>Show Answer</summary>

**Solution:**
```
Number of categories (k) = 4
Dummy variables = k - 1 = 4 - 1 = 3

Columns created: Size_Medium, Size_Large, Size_XL
(Size_Small is the reference/baseline)
```
</details>

---

## 📝 Fill in the Blanks

1. Multiple Linear Regression uses ________ predictors to predict one dependent variable.
   <details><summary>Answer</summary>**two or more (multiple)**</details>

2. VIF stands for ________ ________ ________.
   <details><summary>Answer</summary>**Variance Inflation Factor**</details>

3. If VIF > ________, multicollinearity is considered serious.
   <details><summary>Answer</summary>**10**</details>

4. To avoid the dummy variable trap, we include only ________ dummies for k categories.
   <details><summary>Answer</summary>**k-1**</details>

5. R² always ________ (increases/decreases) when adding more variables.
   <details><summary>Answer</summary>**increases (or stays same)**</details>

6. Adjusted R² can ________ when useless variables are added.
   <details><summary>Answer</summary>**decrease**</details>

7. The formula for VIF is 1 / (1 - ________).
   <details><summary>Answer</summary>**R²**</details>

8. ________ regression adds a penalty λ×Σβ² to handle multicollinearity.
   <details><summary>Answer</summary>**Ridge**</details>

9. The assumption of constant variance of residuals is called ________.
   <details><summary>Answer</summary>**Homoscedasticity**</details>

10. ________ plot is used to check normality of residuals.
    <details><summary>Answer</summary>**Q-Q (Quantile-Quantile)**</details>

---

## ⚡ Shortcuts & Memory Tricks

### VIF Thresholds
**"5 10 Remove!"**
- VIF < 5 = Good
- VIF 5-10 = Watch
- VIF > 10 = Remove!

### Assumptions: L-I-H-N-M
- **L**inearity
- **I**ndependence of errors
- **H**omoscedasticity
- **N**ormality of residuals
- **M** = No **M**ulticollinearity

### One-Hot: "K minus 1 rule"
- K categories → K-1 dummies
- Always remember: "One Less Than Count"

### R² vs Adjusted R²
- R² = "Real estate" - always goes UP with more rooms
- Adj R² = "Adjusted rent" - may go DOWN if rooms are useless

### Coefficient Interpretation
**"All Else Constant"**
β₁ = change in Y per unit change in X₁, **holding all other Xs constant**

### Multicollinearity Detection Order
**"Correlation First, VIF Next"**
1. Quick check: Correlation matrix (|r| > 0.7?)
2. Confirm: VIF calculation

### Normal vs MLR
**"1 vs Many"**
- Simple LR: 1 predictor
- Multiple LR: 2+ predictors
- Everything else is same!

---

## 🎯 Interview Rapid-Fire Answers

| Question | 5-Second Answer |
|----------|-----------------|
| "What is MLR?" | Regression with 2+ predictors |
| "VIF meaning?" | Variance Inflation Factor |
| "Good VIF?" | Less than 5 |
| "One-hot vs Label?" | One-hot for nominal, Label for ordinal |
| "R² vs Adj R²?" | Adj penalizes useless variables |
| "When Adj R² drops?" | When new variable adds no value |
| "Fix multicollinearity?" | Remove, combine, PCA, or Ridge |
| "Dummy trap?" | All k dummies create perfect corr |
| "What's assuming constant variance?" | Homoscedasticity |
| "Unique MLR assumption?" | No multicollinearity |

---

## ✅ Validation Checklist

Before your exam, verify you can:

- [ ] Write the MLR equation
- [ ] Calculate VIF from R²
- [ ] Calculate Adjusted R² from R², n, p
- [ ] Explain why coefficient signs might flip
- [ ] Create one-hot encoded columns (k-1 rule)
- [ ] Interpret regression coefficients
- [ ] List all 5 assumptions
- [ ] Explain when to use Adjusted R² over R²
- [ ] Describe multicollinearity detection methods
- [ ] Explain Ridge vs Lasso for multicollinearity

**All the best! 🎓**
