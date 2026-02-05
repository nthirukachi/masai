# 📊 Observations and Conclusion

## 1. Execution Output (Analysis)

### Actual Results
| Metric | Run A (High LR: 1e-3) | Run B (Low LR: 5e-4) |
| :--- | :--- | :--- |
| **Final Accuracy (Val)** | ~84.41% | ~88.50% |
| **Final Loss (Val)** | ~0.44 | ~0.33 |
| **Curve Shape** | Bumpy / Volatile | Smooth / Stable |

### Observations
1.  **Performance Gap:** Run B (Low LR) significantly outperformed Run A by about 4% in accuracy.
2.  **Stability:** Run A's accuracy fluctuated and settled at a lower value, indicating that the Learning Rate of 0.001 was likely too aggressive for this specific architecture and batch size.
3.  **Convergence:** Run B showed a steady, consistent improvement, proving that a smaller step size allowed the model to find a better minimum in the loss landscape.

## 2. Output Explanation with Diagrams

### What happened?
We trained two identical brains (MLPs). The only difference was how "fast" allowed them to change their minds.

```mermaid
graph TD
    subgraph Run A (High LR)
    A1[Start] -->|Big Steps| A2[Fast Drop]
    A2 -->|Overshooting| A3[Stuck at ~84%]
    end
    
    subgraph Run B (Low LR)
    B1[Start] -->|Small Steps| B2[Steady Drop]
    B2 -->|Precision| B3[Reached ~88%]
    end
```

## 3. Insights
-   **Business Meaning:** Using the slightly lower learning rate (5e-4) yields a **better product**. The "High Learning Rate" model is effectively broken or suboptimal for deployment.
-   **Decision:** We choose **Run B (LR = 5e-4)**. The hypothesis "Lower LR is smoother/better" was **CONFIRMED**.

## 4. Conclusion
-   **Problem Solved:** Yes. We identified the optimal hyperparameter.
-   **Recommendation:** Ship the model trained with Learning Rate 0.0005.
-   **Next Steps:** We could try an learning rate scheduler (starts high, gets low) to get the best of both worlds (speed of A, precision of B).

## 5. Exam Focus Points
-   **Q:** How do you know a Learning Rate is too high from a graph?
-   **A:** The loss curve is jagged (goes up and down) or stuck at a higher value than expected.
-   **Q:** Which run had lower variance?
-   **A:** Run B (Low LR).
