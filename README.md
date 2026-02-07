# Precog Recruitment Task 2026: AI Detection & Adversarial Evolution

## Overview
This repository contains my solution for the "AI Detection & Adversarial Attack" theme. The project moves from identifying 19th-century authors to building a detector that separates human writing from AI generation, explaining its decisions, and finally breaking it with an "Imposter" system.

## Repository Structure
- **Task_0_DataGen.ipynb**: Data collection, cleaning, and preparation pipeline.
- **Task1_Fingerprint.ipynb**: Stylometric analysis distinguishing roughly between Austen and Melville.
- **Task2_Detector.ipynb**: The core detector. Trained on Human (Novels) vs. AI (Gemini) text.
- **Task3_Explainability.ipynb**: Using SHAP to visualize why the detector flags certain paragraphs as AI.
- **Task4_Adversarial.ipynb**: A genetic algorithm that evolves AI text until it fools the detector (The Super-Imposter) and a test on my own Research Statement.

## Setup & Dependencies
To run these notebooks, you'll need a Python environment with PyTorch.
**Dependencies:** `transformers`, `peft`, `shap`, `llama-index`, `pandas`, `scikit-learn`.

*Note for macOS Users:* The notebooks include a fix (`GRPC_DNS_RESOLVER="native"`) to handle local DNS issues with Google's API client.

## Experiments & Inferences

### 1. Styles are Distinct
I found that simple metrics (vocabulary richness, sentence depth) easily separate Austen from Melville. The "Human" signal in this dataset is strongly correlated with 19th-century sentence structures.

### 2. Detection is Easy
My Class C detector (DistilBERT + LoRA) achieved 100% accuracy. This suggests the specific "Human" dataset (classic novels) is so distinct from modern "Generic AI" writing that the problem became trivial for a Transformer.

### 3. The "Super-Imposter" (Adversarial Success)
Using a custom Genetic Algorithm, I evolved a generic AI paragraph (initially 1% Human score) into a "Super-Imposter" scoring **85.07% Human**.
*   **Method:** I used an approach to respect strict API limits (20 calls/day), selecting only the single best survivor and mutating it just 3 times per generation.
*   **Result:** The AI learned to mimic the archaic, dense style of the training data (e.g., "vital frame," "solemn demands"). It didn't become more "human" in a general sense; it became a Victorian novelist.

### 4. The Personal Test (Domain Shift)
The detector classified my modern academic Research Statement as AI (10% Human) which I evolved upto 24%(Human) after mutating paragraph. This confirms the model isn't detecting "humanity"—it's detecting "19th-century fiction style." Modern academic writing falls outside its training distribution, so it fails by default.
