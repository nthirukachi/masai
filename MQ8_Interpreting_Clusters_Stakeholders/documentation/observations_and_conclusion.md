# Observations and Conclusion

## Execution Output
(See `outputs/cluster_profile.csv` and `outputs/cluster_pca.png`)

### Cluster Profile Summary (Typical Results)
| Cluster | Count | Percent | Avg Income (k$) | Avg Score (1-100) | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | ~35 | 17% | ~88 | ~17 | **Savers** (High Earners, Low Spenders) |
| **1** | ~80 | 40% | ~55 | ~50 | **Standard** (Middle Class) |
| **2** | ~39 | 20% | ~86 | ~82 | **Target** (High Earners, Big Spenders) |
| **3** | ~22 | 11% | ~25 | ~79 | **Careless** (Low Earners, Big Spenders) |
| **4** | ~23 | 12% | ~26 | ~20 | **Sensible** (Low Earners, Low Spenders) |

*(Note: Cluster IDs may swap in different runs, but these 5 groups are consistent).*

## Observations
1.  **Distinct Groups:** The data naturally separates into 5 behaviors.
2.  **The "Target" Group:** Approximately 20% of customers are high-value (High Income, High Score). This is the "Gold Mine" for the business.
3.  **The "Standard" Majority:** 40% of customers are average. They are reliable but not the highest impact.
4.  **Risk Group:** The "Careless" group (Low Income, High Spend) might default on credit cards. High risk, high reward.

## Visual Insights
- **PCA Plot:** Shows clear separation, especially for the corner groups (Target, Savers, Careless, Sensible).
- **Central Cluster:** The "Standard" group sits in the middle, overlapping slightly with others, which is expected.

## Business Conclusion
- **Strategy:**
    - **Target Group:** VIP treatment, loyalty programs, new product text alerts.
    - **Savers:** Difficult to convert. Try "Value" messaging or bulk discounts.
    - **Standard:** General mass marketing.
    - **Careless:** Promote installment plans (BNPL), but monitor credit risk.
- **Success:** We successfully translated math (centroids) into a strategy.

## Exam Focus Points
- **Q:** What does the relative size of the 'Standard' cluster tell us?
- **A:** It tells us that most customers are average; if we only focused on the edges, we'd miss the bulk of the volume.
