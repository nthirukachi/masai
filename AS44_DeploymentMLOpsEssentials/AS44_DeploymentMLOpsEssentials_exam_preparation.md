# AS44: Deployment & MLOps Essentials - Exam Preparation

> 📚 **Complete exam revision** for Deployment & MLOps topics
> 🎯 Includes MCQs, MSQs, Numerical Questions, Fill-in-the-Blanks, and Quick Revision

---

## 📋 Topic Coverage Checklist

| Topic | Part | Status |
|-------|------|--------|
| MLOps Fundamentals | Part 1 | ✅ |
| CI/CD for ML | Part 1 | ✅ |
| Safe Deployment (Blue-Green, Canary, Shadow) | Part 1 | ✅ |
| Docker Containerization | Part 2 | ✅ |
| Data Drift & Concept Drift | Part 2 | ✅ |
| Drift Detection (KS Test, Wasserstein) | Part 2 | ✅ |
| Monitoring Without Labels | Part 2 | ✅ |
| Automated Retraining | Part 3 | ✅ |
| Kubernetes Scaling | Part 3 | ✅ |
| Version Control (Code, Data, Model) | Part 3 | ✅ |
| Complete MLOps Pipeline | Part 3 | ✅ |

---

## 📝 Multiple Choice Questions (MCQs)

### Section 1: MLOps Fundamentals

**Q1. What is MLOps?**
- A) Machine Learning Operations - managing ML lifecycle in production
- B) A Python library for ML
- C) A type of neural network
- D) A database for storing models

**Answer:** A

**Explanation:** MLOps = DevOps + Machine Learning. It covers deploying, monitoring, and maintaining ML models in production environments.

---

**Q2. Which statement about ML deployment vs traditional software deployment is TRUE?**
- A) Both are identical in process
- B) ML deployment requires only code testing
- C) ML deployment requires testing code, data, AND model performance
- D) ML deployment is simpler than software deployment

**Answer:** C

**Explanation:** ML has three sources of failure: code, data, and model. All three must be tested before deployment.

---

**Q3. What is a Model Registry?**
- A) A GitHub repository for ML code
- B) Centralized storage for model versions with metadata
- C) A Docker container
- D) A type of database

**Answer:** B

**Explanation:** Model Registry stores trained models with metadata (version, accuracy, hyperparameters) for tracking and rollback.

---

### Section 2: CI/CD for ML

**Q4. In CI/CD, what does CI stand for?**
- A) Continuous Installation
- B) Continuous Integration
- C) Code Improvement
- D) Container Integration

**Answer:** B

**Explanation:** CI = Continuous Integration - automatically testing and building when code changes.

---

**Q5. What is a "Model Gate" in ML deployment?**
- A) Physical gate for server room
- B) Check that blocks deployment if model is worse than baseline
- C) Login portal for model access
- D) Docker container security

**Answer:** B

**Explanation:** Model gates compare new model performance against baseline and reject if worse.

---

**Q6. Which is NOT a typical CI check for ML models?**
- A) Unit tests for preprocessing code
- B) Data schema validation
- C) Model inference latency check
- D) User satisfaction survey

**Answer:** D

**Explanation:** CI checks include code tests, data validation, and model sanity checks - not user surveys.

---

### Section 3: Safe Deployment Strategies

**Q7. In Blue-Green deployment, what do Blue and Green represent?**
- A) Different programming languages
- B) Two separate environments - old and new model
- C) Different teams
- D) Different databases

**Answer:** B

**Explanation:** Blue = current production, Green = new version. Switch traffic completely when ready.

---

**Q8. What is Canary Deployment?**
- A) Deploy to 100% users immediately
- B) Gradually rollout to increasing percentage of users
- C) Deploy only to internal users
- D) Deploy without testing

**Answer:** B

**Explanation:** Canary = gradual rollout (5% → 10% → 25% → 100%), monitoring at each step.

---

**Q9. Which deployment strategy has the LOWEST risk?**
- A) Direct full deployment
- B) Blue-Green
- C) Canary
- D) Shadow

**Answer:** D

**Explanation:** Shadow deployment runs new model in background without affecting users - zero user impact risk.

---

**Q10. In Shadow Deployment, what happens to new model predictions?**
- A) Returned to users
- B) Logged and compared, not returned to users
- C) Discarded immediately
- D) Sent to another server

**Answer:** B

**Explanation:** Shadow model predictions are logged for comparison but never shown to users.

---

### Section 4: Docker Containerization

**Q11. What is Docker?**
- A) A programming language
- B) Platform to package applications with dependencies into containers
- C) A type of database
- D) A cloud provider

**Answer:** B

**Explanation:** Docker packages app + dependencies + environment into portable containers.

---

**Q12. What is the difference between Docker Image and Container?**
- A) They are the same thing
- B) Image is blueprint (static), Container is running instance
- C) Container is blueprint, Image is running instance
- D) Image is for development, Container is for production

**Answer:** B

**Explanation:** Image = template (read-only), Container = running application (active, writable).

---

**Q13. What is Training-Serving Skew?**
- A) Model accuracy difference
- B) Preprocessing mismatch between training and production
- C) Time difference between training and serving
- D) GPU utilization difference

**Answer:** B

**Explanation:** Training-serving skew occurs when preprocessing differs between training and production, causing silent failures.

---

**Q14. Which is a Docker best practice for ML?**
- A) Use unpinned dependencies
- B) Include training and inference in same large image
- C) Pin dependency versions
- D) Store secrets in Dockerfile

**Answer:** C

**Explanation:** Pinned versions ensure reproducibility. Separate train/serve images and never bake secrets.

---

### Section 5: Drift Detection

**Q15. What is Data Drift?**
- A) Model accuracy changes
- B) Input feature distribution changes from training
- C) Code changes
- D) Database migration

**Answer:** B

**Explanation:** Data drift = p(x) shifts - production inputs look different from training inputs.

---

**Q16. What is Concept Drift?**
- A) Input distribution changes
- B) The relationship between input and output changes
- C) Concept is new terminology
- D) Database concept changes

**Answer:** B

**Explanation:** Concept drift = p(y|x) shifts - same input, but correct answer changed.

---

**Q17. Which can be detected WITHOUT ground truth labels?**
- A) Concept drift
- B) Data drift
- C) Model accuracy
- D) F1 score drop

**Answer:** B

**Explanation:** Data drift compares input distributions - no labels needed. Concept drift needs labels.

---

**Q18. What does KS Test (Kolmogorov-Smirnov Test) do?**
- A) Tests model accuracy
- B) Compares if two distributions are different
- C) Tests code correctness
- D) Measures CPU usage

**Answer:** B

**Explanation:** KS test compares training vs production distributions, returns p-value < 0.05 if different.

---

**Q19. What is Wasserstein Distance?**
- A) Physical distance between servers
- B) Measure of how much two distributions differ
- C) Network latency
- D) Model training time

**Answer:** B

**Explanation:** Wasserstein distance measures magnitude of drift between distributions.

---

### Section 6: Monitoring

**Q20. Which is monitored WITHOUT labels in production?**
- A) Accuracy
- B) F1 Score
- C) Input data distribution
- D) Precision

**Answer:** C

**Explanation:** Without labels: monitor data quality, distributions, prediction behavior. With labels: accuracy, F1, etc.

---

**Q21. What is Slice Monitoring?**
- A) Monitoring overall accuracy only
- B) Monitoring performance on subgroups separately
- C) Monitoring time slices
- D) Monitoring storage slices

**Answer:** B

**Explanation:** Slice monitoring checks performance per subgroup - overall average can hide problems.

---

**Q22. Why is monitoring class proportions important?**
- A) For storage planning
- B) Sudden shifts may indicate drift
- C) For CPU usage
- D) For network bandwidth

**Answer:** B

**Explanation:** If predictions suddenly shift (80% Class A instead of 50%), it may indicate data or model issues.

---

### Section 7: Scaling & Kubernetes

**Q23. What is Horizontal Scaling?**
- A) Making server bigger
- B) Adding more servers
- C) Adding more storage
- D) Adding more RAM

**Answer:** B

**Explanation:** Horizontal = more machines. Vertical = bigger machine.

---

**Q24. What is Kubernetes?**
- A) A programming language
- B) Container orchestration platform
- C) A database
- D) A type of Docker

**Answer:** B

**Explanation:** Kubernetes manages, scales, and heals containers automatically in production.

---

**Q25. What is a Pod in Kubernetes?**
- A) A physical server
- B) Smallest deployable unit - one or more containers
- C) A database table
- D) A configuration file

**Answer:** B

**Explanation:** Pod = smallest unit in K8s, contains one or more containers that share resources.

---

**Q26. What does Horizontal Pod Autoscaler (HPA) do?**
- A) Manually scales pods
- B) Automatically adjusts pod count based on metrics
- C) Deletes old pods
- D) Creates pod documentation

**Answer:** B

**Explanation:** HPA automatically scales pods up/down based on CPU, memory, or custom metrics.

---

### Section 8: Version Control & Retraining

**Q27. Which tool is best for DATA versioning?**
- A) Git alone
- B) DVC (Data Version Control)
- C) GitHub
- D) Word documents

**Answer:** B

**Explanation:** Git is for code. DVC tracks large data files with Git-like versioning.

---

**Q28. What does MLflow do?**
- A) Container orchestration
- B) Experiment tracking and model registry
- C) Data storage
- D) Code editing

**Answer:** B

**Explanation:** MLflow tracks experiments (hyperparams, metrics) and manages model registry.

---

**Q29. Which is the SAFEST retraining strategy for adapting to concept drift?**
- A) Full retrain on all historical data
- B) Sliding window (recent N months only)
- C) No retraining ever
- D) Random sampling

**Answer:** B

**Explanation:** Sliding window focuses on recent relevant data, better for adapting to changed patterns.

---

**Q30. What should happen if retrained model is WORSE than baseline?**
- A) Deploy anyway
- B) Reject and keep current model
- C) Delete current model
- D) Increase traffic

**Answer:** B

**Explanation:** Model gates should reject worse models - never deploy a regression.

---

---

## ✅ Multiple Select Questions (MSQs)

**MSQ1. Which are valid safe deployment strategies? (Select ALL that apply)**
- [ ] A) Blue-Green Deployment
- [ ] B) Canary Release
- [ ] C) Shadow Deployment
- [ ] D) Full Direct Deployment

**Answer:** A, B, C

**Explanation:** Blue-Green, Canary, and Shadow are safe strategies. Full direct deployment is risky.

---

**MSQ2. Which are components of MLOps? (Select ALL that apply)**
- [ ] A) Model Versioning
- [ ] B) Continuous Monitoring
- [ ] C) Automated Retraining
- [ ] D) Social Media Management

**Answer:** A, B, C

**Explanation:** MLOps includes versioning, monitoring, and retraining - not social media.

---

**MSQ3. Which can cause training-serving skew? (Select ALL that apply)**
- [ ] A) Different preprocessing code in production
- [ ] B) Different library versions
- [ ] C) Different normalization parameters
- [ ] D) Same identical environment

**Answer:** A, B, C

**Explanation:** Skew is caused by mismatches - same environment means no skew.

---

**MSQ4. Which metrics can be monitored WITHOUT labels? (Select ALL that apply)**
- [ ] A) Input data distribution
- [ ] B) Missing value rates
- [ ] C) Prediction confidence distribution
- [ ] D) F1 Score

**Answer:** A, B, C

**Explanation:** F1 Score requires labels. Other metrics use only input/output data.

---

**MSQ5. Which are Docker best practices? (Select ALL that apply)**
- [ ] A) Pin dependency versions
- [ ] B) Keep images small
- [ ] C) Store secrets in Dockerfile
- [ ] D) Separate training and inference images

**Answer:** A, B, D

**Explanation:** Never store secrets in Dockerfile - use environment variables.

---

**MSQ6. Which should be versioned in MLOps? (Select ALL that apply)**
- [ ] A) Code
- [ ] B) Data
- [ ] C) Model
- [ ] D) Hyperparameters

**Answer:** A, B, C, D

**Explanation:** All four should be versioned for reproducibility.

---

**MSQ7. Which are valid Kubernetes components? (Select ALL that apply)**
- [ ] A) Pod
- [ ] B) Deployment
- [ ] C) Service
- [ ] D) Compiler

**Answer:** A, B, C

**Explanation:** Pod, Deployment, Service are K8s concepts. Compiler is not.

---

**MSQ8. When should you rollback a model? (Select ALL that apply)**
- [ ] A) Sudden performance drop
- [ ] B) High error rates in canary
- [ ] C) Latency exceeds threshold
- [ ] D) Model accuracy improves

**Answer:** A, B, C

**Explanation:** Rollback when things go wrong, not when accuracy improves.

---

---

## 🔢 Numerical Questions

**N1.** A Canary deployment starts at 5% traffic. If each stage doubles the traffic, how many stages before reaching approximately 100%?

**Answer:** 5 stages (5% → 10% → 20% → 40% → 80% → 100%)

---

**N2.** Training accuracy is 95%, production accuracy after 1 month is 80%. What is the percentage drop?

**Answer:** 15 percentage points (or 15.8% relative drop)

**Calculation:** 95% - 80% = 15 percentage points

---

**N3.** KS Test returns p-value = 0.01. Is drift detected at significance level 0.05?

**Answer:** YES (0.01 < 0.05, so drift is significant)

---

**N4.** A model takes 50ms average inference. Kubernetes HPA scales at 70% CPU. If current load is 150 requests/second and each request uses 1% CPU, how many pods needed?

**Answer:** 3 pods

**Calculation:** 150 requests × 1% = 150% CPU needed. At 70% per pod: 150/70 ≈ 2.14 → round up to 3 pods

---

**N5.** Sliding window uses 6 months of data. If data is collected daily for 2 years, what percentage is used?

**Answer:** 25%

**Calculation:** 6 months / 24 months = 0.25 = 25%

---

**N6.** Model registry has 5 versions. Each version is 200 MB. Total storage needed?

**Answer:** 1 GB (5 × 200 MB = 1000 MB)

---

**N7.** A Docker image is 500 MB compressed. If pushed to 3 regional registries, total storage?

**Answer:** 1.5 GB (500 MB × 3)

---

**N8.** Concept drift causes 2% accuracy drop per week. Starting at 95%, when does accuracy fall below 80%?

**Answer:** 8 weeks

**Calculation:** 95% - 80% = 15%; 15% / 2% per week = 7.5 weeks → 8 weeks

---

---

## 📝 Fill in the Blanks

**F1.** The two main types of model drift are _______ drift and _______ drift.

**Answer:** Data, Concept

---

**F2.** In Docker, a _______ is a read-only template, while a _______ is a running instance.

**Answer:** Image, Container

---

**F3.** _______ deployment gradually rolls out to increasing percentages of users.

**Answer:** Canary

---

**F4.** The difference in preprocessing between training and production is called _______ skew.

**Answer:** Training-Serving

---

**F5.** _______ is a statistical test that compares if two distributions are different.

**Answer:** KS Test (or Kolmogorov-Smirnov Test)

---

**F6.** _______ scaling adds more servers, while _______ scaling makes servers bigger.

**Answer:** Horizontal, Vertical

---

**F7.** In Kubernetes, the smallest deployable unit is called a _______.

**Answer:** Pod

---

**F8.** _______ tracks experiments, hyperparameters, and metrics during ML training.

**Answer:** MLflow (or Experiment Tracking)

---

**F9.** _______ deployment runs the new model in background without affecting users.

**Answer:** Shadow

---

**F10.** A _______ _______ blocks deployment if new model is worse than baseline.

**Answer:** Model Gate

---

---

## ⚡ Quick Revision Points

### MLOps Fundamentals
- MLOps = DevOps + ML lifecycle management
- Key components: Version control, CI/CD, Monitoring, Retraining
- Model Registry stores versions with metadata

### CI/CD for ML
- CI: Code tests + Data validation + Model sanity checks
- CD: Model gates + Canary rollout + Monitoring
- Model Gate: Block if performance drops

### Deployment Strategies
| Strategy | Risk | Rollback Speed |
|----------|------|----------------|
| Blue-Green | Medium | Instant |
| Canary | Low | Fast |
| Shadow | Very Low | N/A |

### Docker
- Image = Blueprint, Container = Running instance
- Training-Serving Skew = Preprocessing mismatch
- Best practices: Pin versions, separate images, no secrets

### Drift Types
| Type | What Changes | Detection |
|------|-------------|-----------|
| Data Drift | Inputs p(x) | No labels needed |
| Concept Drift | Relationship p(y\|x) | Labels needed |

### Monitoring
- Without labels: Data quality, distributions, prediction behavior
- With labels: Accuracy, F1, slice performance
- Slice monitoring catches hidden problems

### Kubernetes
- Pod = Smallest unit
- Deployment = What/how many to run
- HPA = Auto-scales based on metrics
- Horizontal scaling preferred

### Version Control
- Git = Code
- DVC = Data
- MLflow = Experiments + Model Registry

---

## 🎯 Interview Quick Answers

| Question | One-Line Answer |
|----------|-----------------|
| What is MLOps? | DevOps + ML - managing ML lifecycle (deploy, monitor, retrain) |
| Data Drift vs Concept Drift? | Data drift = input changes; Concept drift = meaning changes |
| What is Canary deployment? | Gradual rollout to increasing % of users, monitor at each step |
| What is Docker? | Package app + dependencies into portable containers |
| What is Kubernetes? | Container orchestration - manages, scales, heals containers |
| What is training-serving skew? | Preprocessing mismatch between training and production |
| How to detect drift without labels? | Monitor data quality, distributions, prediction behavior |
| What is a Model Gate? | Check that blocks deployment if model is worse than baseline |
| Horizontal vs Vertical scaling? | Horizontal = more machines; Vertical = bigger machine |
| What is MLflow? | Experiment tracking + model registry for ML |

---

## 🚀 Exam Day Shortcuts

1. **Drift Detection**: KS test p < 0.05 = drift exists
2. **Canary Pattern**: 5% → 10% → 25% → 50% → 100%
3. **Docker Rule**: Image is recipe, Container is cake
4. **Scaling Rule**: Horizontal for reliability, Vertical for simplicity
5. **Monitoring Without Labels**: DDDP (Data quality, Distribution, Detection, Prediction behavior)
6. **Version Everything**: C-D-M-C (Code, Data, Model, Config)
7. **Safe Rollout**: Shadow → Canary → Blue-Green (increasing risk)
8. **K8s Basics**: Pod (smallest) → Deployment (how many) → Service (network)

---

> 📘 **Related Materials:**
> - [Part 1: Fundamentals](./AS44_DeploymentMLOpsEssentials1.md)
> - [Part 2: Docker & Drift](./AS44_DeploymentMLOpsEssentials2.md)
> - [Part 3: Scaling & Implementation](./AS44_DeploymentMLOpsEssentials3.md)
