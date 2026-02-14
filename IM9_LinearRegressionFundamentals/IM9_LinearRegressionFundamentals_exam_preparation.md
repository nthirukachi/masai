# IM9: Linear Regression Fundamentals - Exam Preparation

> 📚 **Exam Preparation Guide** for Linear Regression Fundamentals
> 📊 Includes: MCQs, MSQs, Numerical Problems, Shortcuts, Quick Revision

---

## Section 1: Multiple Choice Questions (MCQs)

### Easy Level (1-5)

**Q1.** What type of variable does Linear Regression predict?
- A) Categorical
- B) Binary
- C) Continuous ✅
- D) Ordinal

**Explanation:** Linear Regression predicts continuous numerical values (e.g., price, temperature, sales). Classification predicts categories.

---

**Q2.** In the equation Y = β₀ + β₁X, what does β₀ represent?
- A) Slope
- B) Intercept ✅
- C) Error term
- D) Coefficient of determination

**Explanation:** β₀ is the intercept - the value of Y when X = 0 (baseline value).

---

**Q3.** What does OLS stand for?
- A) Optimal Learning System
- B) Ordinary Least Squares ✅
- C) Overall Linear Statistics
- D) Observed Linear Slope

**Explanation:** OLS minimizes the sum of squared errors between actual and predicted values.

---

**Q4.** Which metric measures the percentage of variance explained by the model?
- A) RMSE
- B) MAE
- C) R-squared ✅
- D) MAPE

**Explanation:** R² (Coefficient of Determination) indicates what % of variance in Y is explained by the model.

---

**Q5.** What is the range of R²?
- A) -∞ to +∞
- B) -1 to +1
- C) 0 to 1 ✅
- D) 0 to 100

**Explanation:** R² ranges from 0 (no variance explained) to 1 (all variance explained).

---

### Medium Level (6-10)

**Q6.** If RMSE = 5 for a house price model in lakhs, what does it mean?
- A) Model explains 5% variance
- B) Average prediction error is ₹5L ✅
- C) 5 features are used
- D) 5 data points were wrong

**Explanation:** RMSE is in the same units as Y. RMSE = 5 means average error magnitude is ₹5L.

---

**Q7.** Which assumption is violated when errors increase with X values?
- A) Linearity
- B) Normality
- C) Homoscedasticity ✅
- D) Independence

**Explanation:** Heteroscedasticity (violation of homoscedasticity) occurs when error variance changes with X.

---

**Q8.** In sklearn, what method is used to train a Linear Regression model?
- A) train()
- B) learn()
- C) fit() ✅
- D) build()

**Explanation:** `.fit(X, y)` trains the model by finding optimal β₀ and β₁ values.

---

**Q9.** Why do we square errors in OLS instead of using absolute values?
- A) Squares are smaller
- B) Prevents cancellation and penalizes large errors more ✅
- C) Mathematical complexity
- D) Industry standard only

**Explanation:** Squaring prevents +/- errors from canceling and penalizes large errors proportionally more.

---

**Q10.** Which library provides detailed statistical summary including p-values?
- A) sklearn
- B) numpy
- C) statsmodels ✅
- D) pandas

**Explanation:** statsmodels provides full statistical output including p-values, confidence intervals, F-statistics.

---

### Hard Level (11-15)

**Q11.** If β₁ = 2.5 in a Sales vs Marketing model, what does it mean?
- A) 25% increase in sales
- B) ₹2.5L sales increase per ₹1L marketing spend ✅
- C) 2.5 is the R² value
- D) 2.5% error rate

**Explanation:** β₁ represents change in Y per unit change in X. ₹1L marketing → ₹2.5L sales increase.

---

**Q12.** What does a p-value < 0.05 for a coefficient indicate?
- A) Coefficient is zero
- B) Coefficient is statistically significant ✅
- C) Model is overfitting
- D) Assumption is violated

**Explanation:** P-value < 0.05 means there's less than 5% probability the coefficient is actually zero.

---

**Q13.** RMSE vs MAE: Which statement is TRUE?
- A) MAE > RMSE always
- B) RMSE ≥ MAE always ✅
- C) They are always equal
- D) No relationship exists

**Explanation:** Due to squaring, RMSE ≥ MAE. They're equal only when all errors are identical.

---

**Q14.** F-statistic in regression tests:
- A) If any single coefficient is significant
- B) If the overall model is significant ✅
- C) If residuals are normal
- D) If homoscedasticity holds

**Explanation:** F-statistic tests if the entire model (all coefficients together) is significant.

---

**Q15.** Which transformation helps when relationship is exponential?
- A) Square root
- B) Log transformation ✅
- C) Polynomial
- D) Min-Max scaling

**Explanation:** Log transformation linearizes exponential relationships: log(Y) = β₀ + β₁X.

---

## Section 2: Multiple Select Questions (MSQs)

**Q1.** Which are assumptions of Linear Regression? (Select ALL that apply)
- ✅ A) Linearity
- ✅ B) Independence of errors
- ❌ C) Errors must be zero
- ✅ D) Homoscedasticity
- ✅ E) Normality of residuals

**Explanation:** LINE: Linearity, Independence, Normality, Equal variance. Errors can never be zero!

---

**Q2.** Which metrics are in the same units as Y? (Select ALL)
- ✅ A) RMSE
- ✅ B) MAE
- ❌ C) R²
- ❌ D) MAPE

**Explanation:** RMSE and MAE are in Y units. R² is ratio (0-1), MAPE is percentage.

---

**Q3.** Which are valid outputs of a trained sklearn LinearRegression model? (Select ALL)
- ✅ A) model.coef_
- ✅ B) model.intercept_
- ❌ C) model.p_value
- ❌ D) model.confidence_interval

**Explanation:** sklearn provides coef_ and intercept_ only. For p-values, use statsmodels.

---

**Q4.** Which can help detect assumption violations? (Select ALL)
- ✅ A) Residual plots
- ✅ B) Q-Q plots
- ✅ C) Durbin-Watson test
- ❌ D) Accuracy score

**Explanation:** Residual plots check linearity/homoscedasticity, Q-Q checks normality, D-W checks independence.

---

**Q5.** Marketing Mix Model uses regression to: (Select ALL)
- ✅ A) Find channel impact on sales
- ✅ B) Calculate ROI per channel
- ✅ C) Optimize budget allocation
- ❌ D) Predict customer churn (classification)

**Explanation:** MMM uses regression coefficients to measure impact, calculate ROI, and optimize budgets.

---

## Section 3: Numerical Problems

### Problem 1: OLS Calculation
**Given:** X = [1, 2, 3, 4, 5], Y = [3, 5, 7, 9, 11]  
**Find:** β₀ and β₁

**Solution:**
```
X̄ = (1+2+3+4+5)/5 = 3
Ȳ = (3+5+7+9+11)/5 = 7

β₁ = Σ(X-X̄)(Y-Ȳ) / Σ(X-X̄)²
   = [(1-3)(3-7) + (2-3)(5-7) + (3-3)(7-7) + (4-3)(9-7) + (5-3)(11-7)] / [(1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²]
   = [(-2)(-4) + (-1)(-2) + (0)(0) + (1)(2) + (2)(4)] / [4 + 1 + 0 + 1 + 4]
   = [8 + 2 + 0 + 2 + 8] / 10
   = 20 / 10 = 2

β₀ = Ȳ - β₁X̄ = 7 - 2(3) = 7 - 6 = 1

Answer: Y = 1 + 2X
```

---

### Problem 2: R² Calculation
**Given:** Actual Y = [10, 20, 30], Predicted Ŷ = [12, 18, 32]  
**Find:** R²

**Solution:**
```
Ȳ = (10+20+30)/3 = 20

SS_total = (10-20)² + (20-20)² + (30-20)²
         = 100 + 0 + 100 = 200

SS_residual = (10-12)² + (20-18)² + (30-32)²
            = 4 + 4 + 4 = 12

R² = 1 - (SS_res / SS_total)
   = 1 - (12/200) = 1 - 0.06 = 0.94

Answer: R² = 0.94 (94% variance explained)
```

---

### Problem 3: RMSE and MAE
**Given:** Actual = [100, 200, 300], Predicted = [90, 210, 280]  
**Find:** RMSE and MAE

**Solution:**
```
Errors: [100-90, 200-210, 300-280] = [10, -10, 20]

MAE = (|10| + |-10| + |20|) / 3 = (10+10+20)/3 = 13.33

MSE = (10² + 10² + 20²) / 3 = (100+100+400)/3 = 200
RMSE = √200 = 14.14

Answer: MAE = 13.33, RMSE = 14.14
```

---

### Problem 4: MAPE Calculation
**Given:** Actual = [50, 100], Predicted = [45, 110]  
**Find:** MAPE

**Solution:**
```
%Error₁ = |50-45|/50 × 100 = 10%
%Error₂ = |100-110|/100 × 100 = 10%

MAPE = (10 + 10) / 2 = 10%

Answer: MAPE = 10%
```

---

### Problem 5: Coefficient Interpretation
**Given:** Sales = 50 + 1.5×TV + 2.0×Social  
**Find:** Which channel has better ROI?

**Solution:**
```
TV: ₹1L spend → ₹1.5L sales → ROI = (1.5-1)/1 × 100 = 50%
Social: ₹1L spend → ₹2.0L sales → ROI = (2.0-1)/1 × 100 = 100%

Answer: Social Media has better ROI (100% vs 50%)
```

---

## Section 4: Quick Shortcuts & Formulas

### Formula Sheet

| Formula | Purpose |
|---------|---------|
| Y = β₀ + β₁X | Regression line |
| β₁ = Cov(X,Y)/Var(X) | Slope calculation |
| β₀ = Ȳ - β₁X̄ | Intercept calculation |
| R² = 1 - SS_res/SS_tot | Coefficient of determination |
| RMSE = √(Σ(Y-Ŷ)²/n) | Root mean squared error |
| MAE = Σ|Y-Ŷ|/n | Mean absolute error |
| MAPE = Σ|Y-Ŷ|/Y × 100/n | Mean absolute % error |

### Quick Memory Tricks

1. **LINE** = Assumptions (Linearity, Independence, Normality, Equal variance)
2. **RMSE ≥ MAE** always (squaring makes it bigger)
3. **R² = 0.85** means model explains 85% (just multiply by 100!)
4. **P-value < 0.05** = Significant (less than 5% chance it's by accident)
5. **β₁ positive** = X↑ then Y↑ (direct relationship)
6. **β₁ negative** = X↑ then Y↓ (inverse relationship)

---

## Section 5: One-Page Quick Revision

### What is Linear Regression?
- Predicts **continuous values** using straight-line relationship
- Equation: **Y = β₀ + β₁X + ε**
- Uses **OLS** to minimize squared errors

### Key Components
- **β₀ (Intercept):** Y when X = 0
- **β₁ (Slope):** Change in Y per unit X
- **ε (Error):** Actual - Predicted

### 4 Assumptions (LINE)
1. **L**inearity - straight line relationship
2. **I**ndependence - errors not correlated
3. **N**ormality - errors follow bell curve
4. **E**qual variance - constant spread

### Metrics to Know
| Metric | Range | Meaning |
|--------|-------|---------|
| R² | 0-1 | % variance explained |
| RMSE | 0-∞ | Error in Y units |
| MAE | 0-∞ | Absolute error |
| MAPE | 0-∞ | % error |

### sklearn Code
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Interview Must-Know
1. Regression = continuous, Classification = categories
2. R² can be negative (worse than mean!)
3. RMSE ≥ MAE always
4. P-value < 0.05 = significant
5. Marketing Mix uses β coefficients for ROI

---

Good luck with your exams! 🎓✨
