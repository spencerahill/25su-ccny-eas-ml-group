# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a summer 2025 CCNY EAS machine learning working group repository focused on learning PyTorch and applying ML models to earth science problems. The emphasis is on understanding both ML concepts and practical implementation.

## Environment Setup

The repository uses conda environments with PyTorch.  The only nonstandard thing is that pytorch needs numpy<2.


## Repository Structure

- `docs/`: Meeting notes, ML concept guides, and example environment files
- `projects/`: Individual member projects with specific ML implementations
  - `michelle/`: Hadley cell edge detection using feedforward neural networks
  - `nielsen-handwriting-nn/`: MNIST digit recognition implementation
  - `spencer/`: NYC precipitation ML model

## Key Concepts

The repository emphasizes understanding:
- Supervised vs unsupervised learning
- Train/validate/test splits and cross-validation
- Data normalization and standardization
- Neural network components (neurons, layers, weights, biases)
- Gradient descent and backpropagation
- Overfitting prevention strategies

## Coding Style Guidelines

For Spencer's project and similar pedagogical implementations:

**Structure:**
- Single script for initial implementations (avoid premature modularization)
- Encapsulate functionality into clear functions
- Use classes only when necessary
- Keep it simple - avoid scope creep and "while we're at it" additions
- Implement precisely what's specified, nothing more, nothing less

**Style:**
- Follow standard Python naming conventions (snake_case)
- Use Black formatting
- Minimal but clear docstrings
- Type hints where helpful
- One function = one clear purpose
- Wrap main execution in `if __name__ == "__main__":`

**Imports:**
- Import only what you need
- No unnecessary dependencies

## Common Workflow

1. Load and explore data in Jupyter notebooks
2. Implement preprocessing scripts
3. Create custom Dataset classes
4. Define model architecture
5. Train with proper validation
6. Evaluate on test set

## Version Control Best Practices

- Git commit after every nontrivial edit. Commit messages should follow best practices:
  - Short summary line ~<80 chars
  - Break with more descriptive summary if needed
  - Additional info in a footer if further needed
  - Upon each git commit, push to remote