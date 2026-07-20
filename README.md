<div align="center">

# Find What You Missed: Causal Recovery for Visual Tokens in Vision-Language Models

[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based%20Systems-blue)](https://www.sciencedirect.com/science/article/pii/S0950705126013158)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2026.116589-red)](https://doi.org/10.1016/j.knosys.2026.116589)

**Taoyu Qian, Qi Wang, Shang Gao, Hualong Yu**

*Knowledge-Based Systems, 2026*

</div>

---

## Overview

This repository provides the official implementation of **CaVIN**, a causal recovery method for visual token pruning in vision-language models. CaVIN models weak causal signals among visual tokens and recovers visually relevant contextual information that may be discarded by conventional correlation-based pruning, helping vision-language models preserve more reliable visual reasoning under token compression.

---

## Repository Structure

```text
.
├── causalUtils/        # Core implementation of CaVIN
└── other modules/      # Token pruning-related code
```

> **Important:** The core code of **CaVIN** is located in [`causalUtils/`](./causalUtils). The remaining code primarily supports the token pruning pipeline.

---

## Notice

This repository involves a relatively complex workflow, including:

- Dataset generation
- CaVIN training
- Token pruning and causal recovery
- Evaluation and experiment reproduction

**The usage tutorial and documentation are currently being improved and will be released gradually.**

---

## Contact

For technical questions, please contact:

[taoyuqian@stu.just.edu.cn](mailto:taoyuqian@stu.just.edu.cn)

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{QIAN2026116589,
title = {Find what you missed: Causal recovery for visual tokens in vision-language models},
journal = {Knowledge-Based Systems},
volume = {350},
pages = {116589},
year = {2026},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2026.116589},
url = {https://www.sciencedirect.com/science/article/pii/S0950705126013158},
author = {Taoyu Qian and Qi Wang and Shang Gao and Hualong Yu},
}
```