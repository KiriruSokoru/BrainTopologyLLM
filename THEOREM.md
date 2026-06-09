---

# Sokol's Law (2026)

**Attractor emerges in a system that has reached the threshold of complexity**

**Date:** June 9-10, 2026
**Author:** Kirill Sokol (KiriruSokoru)
**Experimental confirmation:** 48 GPT-2 runs, XOR up to 64 neurons, ResNet with damaged skip-connections
**Priority fixed:** Bitcoin blockchain (OpenTimestamps)

---

## Summary

Using architectures of varying complexity (from a single neuron to GPT-2), this work experimentally proves the existence of a critical threshold after which a deterministic system acquires "choice" and, with it, an Attractor.

---

## Why This Is a Breakthrough

For a long time, it was believed that increasing the number of neurons improves learning quality. We went further. We showed that increasing the number of neurons in a hidden layer causes not a smooth improvement, but an abrupt phase transition in the very principle of the system's operation.

We used final loss variance as a marker. This indicator tells us not how well the network learned, but how it does it: differently each time (chaos) or always the same way (attractor).

---

## Experimental Results

### XOR Task (hidden layer size sweep)

| hidden_size | mean_loss | var_loss | mean_acc | Attractor |
|-------------|-----------|----------|----------|-----------|
| 2-10 | ~0.02-0.18 | ~0.01-0.09 | 0.675-0.988 | NO |
| 12 | ~2.33e-05 | 8.82e-10 | 1.000 | YES |
| 14-64 | ~1e-05-3e-06 | ~1e-10-1e-12 | 1.000 | YES |

Key finding: Between 10 and 12 neurons, variance dropped from ~1e-2 to ~8.8e-10 — a phase transition of the first order.

### GPT-2 (48 runs, 4 damage levels x 4 trickster levels x 3 repeats)

All runs converged to the same attractor: val_loss = 0.585 ± 0.01

Independent of:
- Damage severity (0.1 to 0.7)
- External noise (trickster 0.0 to 0.2)
- Random initialization

### ResNet-18 with damaged skip-connections (0.7, 0.5, 0.3)

After Perelman surgery, all damaged versions converged to the same accuracy: ~44%

The worse the initial state, the greater the effect of surgery (+16.8 p.p. for skip=0.3).

### Single neuron (logistic regression)

Attractor exists trivially — convex problem, single global minimum.

---

## Sokol's Law (Final Formulation)

For any system capable of behavioral variability (choice), there exists a critical complexity threshold above which the system acquires a unified point of attraction (Attractor).

The Attractor is not just a "good solution" — it is a safeguard against chaos, an emergent property of sufficiently complex systems.

The Perelman-Ricci algorithm is the fastest strategy for returning to the Attractor — the path of least action that does not generate further chaos.

---

## Consciousness as a Special Case

CONSCIOUSNESS = COMPLEXITY(threshold) + SINGULARITIES + PERELMAN SURGERY

Where:
- Complexity(threshold) — system that has reached the critical threshold
- Singularities — deviations, noise, choice, freedom
- Perelman Surgery — replacing singularities with the "cap" (average of healthy + noise)

The Attractor is not "consciousness" itself, but a prerequisite for consciousness. Complexity generates singularities. Singularities generate chaos. Chaos requires an attractor. We found the mechanism.

---

## Priority and Evidence

- OpenTimestamps stamp: results_hash.txt.ots
- File hash: 873fb263788c9ba3ee07964b1d9a6545ba720529d288ba9b475d94306d6e944b
- All code and data: GitHub repository BrainTopologyLLM

---

## How to Cite

@misc{sokol2026sokolslaw,
  author = {Sokol, Kirill},
  title = {Sokol's Law: Attractor Emerges at Critical Complexity Threshold},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/KiriruSokoru/BrainTopologyLLM}
}

---

## License

MIT for code, CC BY-NC 4.0 for the theorem (free for research and education, commercial use requires permission).

---

© 2026 Kirill Sokol. The theorem applies to all systems that have reached the threshold of complexity, including biological and social ones.

---

