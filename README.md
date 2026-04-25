# ChineseChessAI-Zero

A high-performance C# implementation of a Chinese Chess (Xiangqi) AI, based on the **AlphaZero** (CCZero) methodology. This project features a deep residual neural network integrated with Monte Carlo Tree Search (MCTS), optimized for modern NVIDIA hardware.

> **Development Note**: This project was architected and implemented by **Gemini** (Google's AI) under the iterative guidance, technical constraints, and domain expertise of the lead developer.

## 🚀 Key Features

* **AlphaZero Framework**: A self-play reinforcement learning pipeline (Self-Play -> Replay Buffer -> Training).
* **Neural Network**: Residual Network (ResNet) architecture with dual heads (Policy and Value), implemented via **TorchSharp**.
* **MCTS Engine**:
* Asynchronous Batch Inference to maximize GPU utilization.
* Support for **3200+ simulations** per move.
* Advanced PUCT (Predictor Upper Confidence Bound applied to Trees) algorithm.


* **Xiangqi Domain Logic**:
* High-speed legal move generation with `IsKingSafe` and "Flying General" (King-to-King) detection.
* Hardcoded **Threefold Repetition** detection and penalty logic.


* **Training Optimizations**:
* Penalty-based draw avoidance (incentivizing aggressive play).
* Support for Large Batch Sizes (4096) and multi-epoch training.



## 🛠️ Tech Stack

* **Language**: C# / .NET 10
* **UI Framework**: WPF (Windows Presentation Foundation)
* **Deep Learning**: [TorchSharp]() (LibTorch 2.x bindings)
* **Hardware Acceleration**: NVIDIA CUDA (Optimized for **RTX 2000 Ada**)

## 📈 Current Status (Iteration 91+)

The model has transitioned from a defensive posture to an aggressive tactical phase:

* **Loss Stability**: Current Loss resides around **1.5 - 1.9**, stabilizing after a major logic update regarding draw penalties.
* **Tactical Depth**: Demonstrates strong defense (up to 1000 moves) and emerging "Kill" instinct (checkmates observed within 70 moves in mid-game).
* **Performance**: Achieving ~49% GPU utilization on RTX 2000 Ada with 3200 MCTS simulations.

## 📋 Configuration & Usage

### Prerequisites

* Windows 10/11
* NVIDIA GPU with **CUDA Toolkit 12.x** installed.
* .NET 10 SDK

### Running the Pipeline

1. **Self-Play**: Launch the application to start generating self-play games. Samples are automatically stored in the `ReplayBuffer`.
2. **Training**: The `Trainer` module will periodically pull samples and perform gradient descent to update the `best_model.pt`.
3. **Inference**: Use the `MCTSEngine` to play against the AI or analyze specific board positions.

## Engineering Notes

These implementation constraints are important for correctness and should be preserved unless the surrounding design changes.

* **MCTS virtual loss sign**: `MCTSNode.GetPUCTValue` must treat virtual loss as making an in-flight child temporarily worse from the parent perspective. With `W` stored in the node-to-move perspective, the correct term is `q = -(W + vl) / (N + vl)`, not `-(W - vl) / (N + vl)`.
* **Side-relative training encoding**: `StateEncoder.Encode` is already side-relative. Plane `0..6` always represents the current player and black-to-move positions are flipped into the same canonical view. Because of that, `180-degree rotation + color swap` is not a valid extra augmentation for PGN/CSV import data and must not be exported as separate mirror samples. If additional augmentation is needed, use a symmetry that actually changes the canonical input, such as left-right mirroring with matching policy remapping.
* **League result strings must stay canonical UTF-8 Chinese**: `SelfPlay` emits `"平局"`, `"红胜"`, and `"黑胜"`, and ELO settlement in `TrainingOrchestrator` must compare against those exact strings. Mojibake variants such as `"骞冲眬"` or `"绾㈣儨"` will never match and will corrupt league ratings. Persist league/game JSON as UTF-8 to avoid reintroducing this failure mode.
* **MCTS concurrent collection access must use snapshots**: When reading `MCTSNode.Children` or the internal bounded caches from multiple search threads, do not run LINQ directly on mutable `ConcurrentDictionary` instances. Take `ToArray()` snapshots first, then sort/select on the snapshot. Direct `OrderBy(...).ToArray()` over the live dictionary can reproduce the `dictionary is greater than the available space` league crash under parallel play.
* **Corrupt agent model files must be quarantined, not retried forever**: `TrainingOrchestrator.GetOrAddAgent` can encounter `EndOfStreamException` while `TorchSharp` loads a truncated `.pt` file. In that case the file should be moved aside as `*.corrupt_*`, the event must be logged with `AgentId` and path, and the agent should continue with a fresh model instead of repeatedly failing matches.
* **`self_play_data` is retired**: Active training data paths are `data/master_data` and `data/league_data`. The old `data/self_play_data` folder is no longer part of the live pipeline and should not be reintroduced as a default storage location.

## 🤝 Acknowledgments

This project is a testament to AI-Human collaboration. The core engine, state encoding, and MCTS optimizations were generated by **Gemini**, while the strategic direction, troubleshooting of customs/shipping logistics for hardware, and domain-specific rules were provided by the user.
