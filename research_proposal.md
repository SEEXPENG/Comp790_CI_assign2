# Research Proposal: Perception-Guided Optimization in Holographic Displays

## Project Information
- **Title**: Perception-Guided Optimization in Holographic Displays: Balancing Quality, Efficiency, and Hardware Constraints
- **Principal Investigators**: Xi Peng UNC Chapel Hill

## 1. Introduction
Holographic displays offer a pathway to highly realistic 3D visualizations, surpassing the limitations of stereoscopic or light field displays. However, their practical deployment remains constrained by:

- Hardware limitations (e.g., bit-depth, spatial light modulators)
- Visual artifacts (e.g., speckle noise)
- Perceptual thresholds of human users

This proposal aims to explore and quantify perceptually important trade-offs in holographic display systems, with a focus on optimizing:
- Bit-depth
- Phase regularization
- Hardware configuration

## 2. Research Objectives

### O1: Perceptual Quality Modeling
- Build a perceptual quality model using Maximum Likelihood Difference Scaling (MLDS)
- Focus on bit-depth impact on visual quality

### O2: Phase Strategy Analysis
- Compare smooth-phase and random-phase holograms
- Evaluate:
  - Visual fidelity
  - Computational demand
  - Speckle suppression

### O3: System Parameter Investigation
- Study impact of:
  - Eyebox size
  - Near-field vs far-field configurations
  - Sparse vs dense 3D scenes
- Focus on perceptual realism

### O4: Hardware Trade-off Evaluation
- Compare LCOS and DMD spatial light modulators
- Analyze:
  - Perceptual efficiency
  - Computational efficiency

## 3. Background and Related Work

### Previous Research Findings
- Phase initialization and smooth-phase regularization are critical for speckle suppression [Chakravarthula et al. 2019, Shi et al. 2022]
- Energy concentration reduces eyebox size [Wang et al. 2024]
- Neural methods improve defocus blur realism [Yang et al. 2022]

### Perceptual Studies
- Bit-depth thresholds [Maloney and Yang 2003]
- Speckle visibility [Georgiou et al. 2023]

## 4. Methodology

### 4.1 Perceptual Modeling via Bit-Depth Scaling
1. Conduct 2AFC (two-alternative forced choice) user study
2. Fit perceptual quality curve using MLDS
3. Determine thresholds for:
   - "Acceptable" quality regions
   - "Diminishing returns" regions

### 4.2 Phase Strategy Comparison
1. Generate holograms:
   - Smooth-phase
   - Random-phase
2. Evaluate artifacts:
   - Speckle
   - Defocus blur
   - Ringing
3. Use both:
   - User studies
   - PSNR/SSIM metrics

### 4.3 System Parameters Evaluation
1. Eyebox Analysis:
   - Vary sizes
   - Measure quality drop-off
2. Content Analysis:
   - Simulate sparse/dense 3D content
   - Visualize reconstructions
3. Field Configuration:
   - Compare near-field/far-field
   - Assess 2D perception accuracy

### 4.4 Hardware Trade-off Analysis
1. Prototype Comparison:
   - LCOS vs DMD
   - Simulation and real-world testing
2. Performance Metrics:
   - Temporal multiplexing impact
   - Perceptual stability
   - Hardware complexity
   - Cost-benefit analysis

## 5. Expected Contributions
1. Validated perceptual model for bit-depth vs quality
2. Practical guidelines for hologram phase optimization
3. Quantitative trade-off maps for system configuration
4. Hardware selection insights based on user-centric criteria

## 6. Potential Impact
This work directly advances the field of perceptual display technology by:
- Establishing concrete guidelines for practical holographic rendering
- Informing academic research
- Guiding industrial product design
- Supporting AR/VR systems requiring realistic 3D visual output under hardware limitations