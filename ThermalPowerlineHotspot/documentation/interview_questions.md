# 🎤 Interview Questions: Thermal Powerline AI Project

## 🧠 Part 1: Conceptual Understanding (Beginner/HR)

#### 1. What is the main goal of this project?
- **Simple Answer:** To find hot spots on power lines using AI so we can fix them before they break.
- **Analogy:** It's like a doctor looking for a fever to stop a cold from becoming pneumonia.

#### 2. Why do we need AI for this? Can't humans look at the images?
- **Answer:** Humans get tired and slow. AI can look at thousands of images in seconds without missing anything.
- **Key Point:** Scalability and consistency.

#### 3. What is a "Thermal Anomaly"?
- **Answer:** A temperature that is significantly higher than normal or its surroundings.
- **Context:** In power lines, heat = wasted energy or bad connection.

#### 4. Why use a Drone?
- **Answer:** Safe (no climbing), fast, and can go where trucks can't.

#### 5. How does the AI know what a "bad" hotspot looks like?
- **Answer:** We trained it on examples of "Good" and "Bad" tiles. It learned the patterns (like high temp + uneven spread).
- **Term:** Supervised Learning.

---

## 🛠️ Part 2: Technical & ML (Data Scientist)

#### 6. Why did you use Random Forest instead of Deep Learning?
- **Technical Answer:** The input was tabular (structured features), not raw images. RF is robust for tabular data, handles outliers well, and gives feature importance.
- **Diagram:**
  ```mermaid
  flowchart LR
  TabularData --> RandomBest
  RawImages --> CNN_Best
  ```

#### 7. How did you handle the class imbalance (Normal >> Anomaly)?
- **Answer:** I used `class_weight='balanced'` in the model and relied on F1-Score/Recall instead of Accuracy.
- **Mistake to Avoid:** "I just used accuracy." (Instant fail).

#### 8. Explan `ROC-AUC` in one sentence.
- **Answer:** It's a score from 0 to 1 that tells us how good the model is at distinguishing between positive (fault) and negative (normal) classes.

#### 9. Which feature was most important?
- **Answer:** `hotspot_fraction` and `temp_mean`.
- **Why:** Absolute temp fluctuates with weather, but the *fraction* of a tile that is hot is a strong structural indicator of a defect.

#### 10. What is "Precision" vs "Recall" in this context?
- **Precision:** "If I call the crew, is it a real emergency?" (Cost efficiency).
- **Recall:** "If there is a fire risk, did I find it?" (Safety).
- **Choice:** We prioritize **Recall**.

#### 11. What is Overfitting and how did you prevent it?
- **Answer:** The model memorizing the training data instead of learning rules.
- **Prevention:** Limited tree depth (`max_depth=10`), minimum samples per leaf (`min_samples_leaf=2`).

#### 12. How does `train_test_split` help?
- **Answer:** It hides some data during training to simulate "future" data. It proves the model works on new cases.

#### 13. What if the ambient temperature changes (Winter vs Summer)?
- **Answer:** That's why we use `delta_to_neighbors`. Even in winter, a faulty wire is hotter than the wire next to it. Absolute temp is less useful than relative temp.

#### 14. How would you deploy this?
- **Answer:** Load the model onto the drone's onboard computer (Edge AI) to flag issues in real-time, or process data in the cloud after landing.

---

## 🚀 Part 3: Advanced / Scenario Based

#### 15. The model is flagging too many false alarms. What do you do?
- **Action:** Increase the decision threshold (e.g., probability > 0.6 instead of 0.5).
- **Trade-off:** You might miss some real faults (Lower Recall).

#### 16. We now have GPS data. How do you improve the project?
- **Idea:** Use GPS to cluster faults. If 5 faults happen at the same GPS coordinate over 3 months, it's a chronic hardware issue, not random.

#### 17. Can we predict *when* it will fail?
- **Answer:** Not with this dataset. We need **Time-Series** data (history of temp changes) to do Predictive Maintenance (Regression). This project is Condition-Based Maintenance (Classification).

#### 18. Why not use simple rules like `if temp > 50 then fail`?
- **Answer:** Too simple. 50°C might be normal in summer but fatal in winter. AI learns complex, non-linear combinations of Load + Temp + Ambient.

#### 19. What if we get 10x more data?
- **Answer:** Random Forest scales well, but we might switch to XGBoost (Gradient Boosting) for slightly better accuracy if training time allows.

#### 20. Explain the Confusion Matrix output.
- **Scenario:** `[[90, 5], [2, 8]]`
- **Explanation:** "We correctly found 90 safe lines and 8 faults. We missed 2 real faults (Dangerous!), and we had 5 false alarms (Waste of time)."
```mermaid
grid
    title Confusion Matrix
    TN "Safe (90)": 90
    FP "False Alarm (5)": 5
    FN "Missed (2)": 2
    TP "Found (8)": 8
```
