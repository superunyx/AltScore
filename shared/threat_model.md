# Threat Model

## 3.0 Feature-layer privacy design
The model's input features are now ratios and rates (dimensionless, scale-invariant). This means no absolute income or expense figures ever exist in the feature vector at all. This is a privacy property independent of and in addition to the Phase 9 FL/DP/encryption work. It acts as a fourth, earlier layer of defense by ensuring the raw financial amounts are not even accessible in the feature representation, without replacing the existing federated learning, differential privacy, and encryption layers.
