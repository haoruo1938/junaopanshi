# Research Proposal: Embodied Navigation Generalization

## Background

Embodied agents must complete language-guided navigation in diverse indoor environments under partial observability.
Real deployment requires robustness to scene shifts, object appearance variation, and ambiguous user commands.

## Goal

Build a language-conditioned embodied navigation agent that:

1. Navigates to user-specified targets (e.g., sofa, bed) in household scenes.
2. Uses only onboard visual observations and robot state.
3. Supports multi-turn interaction with completion feedback.

## Task Definition

- **Input**: language command (text/voice), RGB-D observation, robot state.
- **Output**: low-level navigation actions (`RotateLeft`, `RotateRight`, `MoveAhead`) and dialogue response.
- **Success**: target reached within threshold distance and step budget.
- **Constraint**: no privileged global object coordinates for policy decisions.

## Related Methods

- Vision-language grounding for target understanding.
- Active visual search in partially observed scenes.
- Local reactive control (visual servoing) with depth-based stopping.
- Optional semantic mapping for long-horizon planning.

## Implementation Path

1. MVP: heuristic visual servoing with rule-based target parser.
2. Multi-scene evaluation and failure recovery policies.
3. Plug-in semantic map memory and global planner.
4. Integrate policy learning / RL fine-tuning with intervention data.

## Risks and Mitigations

- **Risk**: target often occluded.
  - **Mitigation**: scan-and-reacquire behavior and retry budget.
- **Risk**: rule parser lacks language coverage.
  - **Mitigation**: add LLM parser fallback and synonym expansion.
- **Risk**: local policy gets stuck.
  - **Mitigation**: add memory-based exploration and backtracking triggers.

## Expected Deliverables

1. Runnable web demo.
2. Open-source repository with setup instructions.
3. Quantitative report (success rate, steps, completion time).
4. Roadmap toward semantic SLAM and cross-environment generalization.
