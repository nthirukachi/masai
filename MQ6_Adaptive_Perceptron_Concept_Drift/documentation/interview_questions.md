# 🎤 Interview Questions

## 1. Concept Drift vs Data Drift
**Q1: What is the difference between Concept Drift and Data Drift?**
- **Simple Answer:** Concept Drift is when the *answer* changes (e.g., bad emails typically have "Free", now they have "Prize"). Data Drift is when the *inputs* change but the answer logic is same (e.g., getting more night-time users, but fraud logic is same).
- **Technical Answer:** Concept Drift is a change in the posterior distribution $P(y|X)$. Data Drift (Covariate Shift) is a change in input distribution $P(X)$ while $P(y|X)$ remains constant.
- **Analogy:**
    - Concept Drift: The exam syllabus changed.
    - Data Drift: The exam questions are just printed in a different font or harder phrasing.

## 2. Adaptive Learning
**Q2: Why do we decay the learning rate?**
- **Simple Answer:** To learn fast at the start, but be careful and precise at the end.
- **Technical Answer:** Large learning rates help escape local minima early but cause oscillation around the global minimum. Decay allows for fine-grained convergence.
- **Analogy:** Parking a car. You drive fast into the lot, but creep inches forward when nearing the wall.

## 3. Reset Mechanism
**Q3: Is resetting weights a good idea in production?**
- **Simple Answer:** Only as a last resort. It's like formatting your computer because it's slow. Effective, but you lose everything.
- **Technical Answer:** It's a "Global Reset". It's drastic. Often, *ensembles* or *sliding windows* are better because they gradually forget old data rather than sudden amnesia.
- **Common Mistake:** Resetting too often (on noise) makes the model useless.

## 4. Perceptron Limitations
**Q4: Why use a Perceptron for this and not a Deep Neural Network?**
- **S:** It's fast and simple for testing concepts.
- **T:** Perceptrons have guaranteed error bounds and are computationally typically $O(d)$ per update, ideal for high-speed value streams.
- **Mistake to avoid:** Thinking Perceptrons can learn complex shapes (XOR). They can't.

## 5. Validation Check
**Q5: Why validate on the *next* batch?**
- **S:** To see if the model is ready for what's coming.
- **T:** In prequential evaluation (Test-Then-Train), every sample is used for testing model performance on detecting drift *before* it is used for training. This gives the most honest drift detection.

## 6. Real World
**Q6: Give a real-world example of mild drift.**
- **Ans:** Seasonal shopping trends. People buy coats in winter, swimsuits in summer. The "likelihood to buy" shifts, but doesn't flip entirely.

## 7. Metrics
**Q7: Which metric is best for drift? Accuracy or F1-Score?**
- **Ans:** F1-Score is usually better if class balance changes (e.g., suddenly way more fraud attempts). Accuracy can be misleading if one class dominates.

## 8. Parameters
**Q8: What happens if `decay_rate` is 0.1 (very small)?**
- **Ans:** The model stops learning almost immediately (one big step, then almost zero). It becomes a static model very fast.

## 9. Hyperparameters
**Q9: How to choose `decay_steps`?**
- **Ans:** Empirical testing. If data changes fast, you might not want to decay too much, or you want to restart the schedule.

## 10. Improvements
**Q10: How would you improve this model?**
- **Ans:** Use ADWIN (Adaptive Windowing) to automatically resize the training window instead of fixed resets.
