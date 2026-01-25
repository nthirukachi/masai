# 📝 Exam Preparation: Thermal Powerline AI Project

## 🅰️ Section A: Multiple Choice Questions (MCQ)

**Q1. What is the primary target variable in this project?**
a) `temp_mean`
b) `hotspot_fraction`
c) `fault_label`
d) `risk_score`
**Answer: c) fault_label**
*Explanation: We are predicting whether a tile is faulty (1) or normal (0).*

**Q2. Which library is used for creating the Random Forest model?**
a) pandas
b) matplotlib
c) seaborn
d) sklearn (scikit-learn)
**Answer: d) sklearn**
*Explanation: `sklearn.ensemble.RandomForestClassifier` is the tool.*

**Q3. If `Recall` is 1.0 (100%), what does it mean?**
a) We found ALL the anomalies.
b) We made no false alarms.
c) The model is perfect.
d) The accuracy is 100%.
**Answer: a) We found ALL the anomalies.**
*Explanation: Recall = TP / (TP + FN). Perfect Recall means FN=0 (no missed faults).*

**Q4. Why do we split data into Train and Test sets?**
a) To make training faster.
b) To evaluate performance on unseen data.
c) To fix bugs in the code.
d) To increase accuracy.
**Answer: b) To evaluate performance on unseen data.**
*Explanation: Testing on training data is "cheating" and leads to overfitting.*

**Q5. What does a `correlation` of 0.85 between two features mean?**
a) They are unrelated.
b) They are strongly negatively related.
c) They are strongly positively related (Redundant).
d) One causes the other.
**Answer: c) Strongly positively related.**
*Explanation: High correlation suggests features carry similar information.*

---

## 🅱️ Section B: Multiple Select Questions (MSQ)

**Q6. Which of the following are valid performance metrics for this classification problem? (Select all that apply)**
a) [x] F1-Score
b) [x] Precision
c) [ ] Mean Squared Error (MSE)
d) [x] Recall
e) [x] ROC-AUC
**Answer: a, b, d, e**
*Explanation: MSE is for regression (predicting numbers), not classification.*

**Q7. What can cause a "False Positive" in thermal inspection? (Select all that apply)**
a) [x] Sun glint on the wire.
b) [ ] Broken wire strand.
c) [x] Bird nest retaining heat.
d) [ ] Loose connection.
**Answer: a, c**
*Explanation: Broken strands and loose connections create REAL heat (True Positives).*

---

## 🔢 Section C: Numerical Problems

**Q8. Calculate Precision.**
- True Positives (TP) = 80
- False Positives (FP) = 20
- False Negatives (FN) = 10

**Solution:**
$$Precision = \frac{TP}{TP + FP}$$
$$Precision = \frac{80}{80 + 20} = \frac{80}{100} = 0.8$$
**Answer: 0.8 (or 80%)**

**Q9. Calculate Recall.**
- True Positives (TP) = 80
- False Positives (FP) = 20
- False Negatives (FN) = 10

**Solution:**
$$Recall = \frac{TP}{TP + FN}$$
$$Recall = \frac{80}{80 + 10} = \frac{80}{90} \approx 0.89$$
**Answer: 0.89 (or 89%)**

---

## 🔡 Section D: Fill in the Blanks

**Q10. The \_\_\_\_\_\_\_\_\_\_ function in pandas is used to read a CSV file.**
**Answer: read_csv**

**Q11. In a Decision Tree, the top-most node is called the \_\_\_\_\_\_\_\_\_\_.**
**Answer: Root Node**

**Q12. \_\_\_\_\_\_\_\_\_\_ is a visualization that shows the correlation between all variables.**
**Answer: Heatmap**

**Q13. To split a dataset into 80% train and 20% test, we set the parameter `test_size` to \_\_\_\_\_\_\_\_\_\_.**
**Answer: 0.2**
