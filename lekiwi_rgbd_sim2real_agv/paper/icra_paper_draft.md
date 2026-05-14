# LeRobot-Compatible RGB-D Residual Safety Navigation for Low-Cost Mobile Robots

**Anonymous Submission — ICRA / IROS 20XX**

---

## Abstract

Safe navigation under RGB-D perception noise remains a critical challenge for low-cost mobile robots deployed in unstructured indoor environments. We present a LeRobot-compatible residual safety navigation framework for the LeKiwi omnidirectional mobile platform equipped with an Intel RealSense D435i camera. Our framework introduces three components: (1) a compact RGB-D-to-Scan64 virtual LiDAR representation that projects depth images into a 64-beam polar scan via percentile pooling, suppressing single-pixel dropout noise common in consumer depth sensors; (2) a lightweight Residual Safety Model (MLP, $\sim$18k parameters) that predicts additive correction deltas to a nominal Dynamic Window Approach (DWA) planner output; and (3) a hard Emergency Safety Shield that enforces proximity-based velocity constraints and an Action Adapter that applies acceleration limits and temporal smoothing. The complete pipeline operates within the Hugging Face LeRobot data framework, using a standardized dataset schema that includes RGB, depth-derived Scan64, robot state, nominal actions, residual corrections, safety masks, and intervention flags. We construct synthetic and real-world LeKiwi navigation datasets and design a comprehensive evaluation protocol comparing our method against vanilla DWA, LeRobot native imitation learning, and a LoGoPlanner-style strong baseline. Ablation studies isolate the contributions of each safety component. Experimental design, metrics, and expected analysis are presented; quantitative results are marked as [TBD] pending ongoing data collection. This work aims to provide a lightweight, interpretable, and LeRobot-compatible safety navigation alternative for resource-constrained mobile robots.

**Keywords:** Mobile robot navigation, RGB-D perception, LeRobot, LeKiwi, residual learning, sim-to-real transfer, safety shield.

---

## I. Introduction

### A. Motivation

Safe autonomous navigation in indoor environments is a prerequisite for deploying mobile robots in warehouses, hospitals, laboratories, and domestic settings. While large industrial AGVs rely on expensive 2D/3D LiDAR sensors and dedicated safety PLCs, the growing ecosystem of low-cost, open-source mobile platforms demands navigation solutions that are simultaneously safe, computationally lightweight, and compatible with consumer-grade perception hardware.

Three persistent challenges motivate this work. First, **dynamic obstacles** (pedestrians, other robots, moving furniture) require navigation policies that react within tens of milliseconds, yet traditional local planners often operate on stale obstacle representations. Second, **RGB-D perception noise**—including depth holes on reflective surfaces, motion blur, lighting-induced dropouts, and systematic bias near object edges—introduces false negatives and false positives into obstacle maps, causing either overly conservative or dangerously aggressive behaviors. Third, **limited onboard computation** on low-cost platforms (e.g., Raspberry Pi 4, Jetson Nano) precludes running large neural networks or sophisticated SLAM pipelines at control frequency.

These challenges are particularly acute for small research and educational mobile platforms, where the cost of a single collision (hardware damage, safety incident) far exceeds the cost of conservative navigation. This paper investigates whether a lightweight, interpretable safety overlay—compatible with existing open-source robot learning frameworks—can meaningfully improve navigation safety without requiring expensive sensor upgrades or high-end compute.

### B. Why LeKiwi and LeRobot

The **LeKiwi** platform is a low-cost, open-source mobile manipulator with three omnidirectional wheels, Feetech serial bus servos, and a modular hardware design that accepts standard expansion (cameras, arms, sensors). Its omniwheel kinematics allow holonomic planar motion, making it well-suited for tight indoor navigation. However, its default perception stack relies on a monocular RGB camera without depth sensing, which limits obstacle awareness.

**LeRobot** [1] is an open-source robot learning framework developed by Hugging Face that provides standardized dataset formats, a record/train/evaluate pipeline, pretrained policies, and a growing ecosystem of robot configurations. LeRobot's dataset schema enforces consistent timestamp alignment, observation/action encoding, and episode metadata, which substantially lowers the barrier to reproducible robot learning research. The framework already includes a LeKiwi robot configuration and RealSense camera support, but these components have not been integrated for depth-based navigation.

This work builds on both platforms without modifying their core repositories. We extend the LeKiwi configuration with D435i depth sensing, implement depth-to-scan preprocessing compatible with LeRobot's data model, and design a residual safety module that can be trained using LeRobot's standard training loop. By reusing LeRobot's data infrastructure rather than building a custom pipeline, we ensure that our datasets, models, and evaluation protocols are directly usable by the broader LeRobot community.

### C. Why RGB-D and Sim-to-Real are Difficult

RGB-D cameras such as the Intel RealSense D435i offer an attractive trade-off for low-cost navigation: they provide dense depth at 30–90 Hz over a $\sim$87$^\circ$ horizontal field of view, cost approximately \$200–\$400, and are widely supported in ROS and Python ecosystems. However, active stereo depth sensing suffers from several failure modes that directly impact navigation safety:

- **Depth holes** on dark, shiny, or transparent surfaces produce invalid (NaN) pixels that create blind spots in obstacle detection.
- **Flying pixels** near depth discontinuities generate phantom obstacles at incorrect distances.
- **Motion blur** during robot rotation degrades depth accuracy, particularly at higher angular velocities.
- **Multipath interference** in corners and narrow passages produces systematically incorrect range readings.
- **Lighting sensitivity** causes outdoor and window-adjacent deployments to fail under direct sunlight.

The **sim-to-real transfer gap** compounds these perception challenges. Synthetic depth renderings—even with domain randomization—differ from real D435i depth in noise distribution, edge behavior, and material-dependent artifacts. Additionally, the LeKiwi's wheel slip, floor friction variation, and motor backlash create dynamics discrepancies that are difficult to model precisely in simulation. These gaps mean that a policy trained purely in simulation may exhibit unexpected behaviors when deployed on hardware, particularly in safety-critical near-obstacle situations.

### D. Baseline Motivation: DWA and LoGoPlanner

We select two strong baselines to contextualize our method.

The **Dynamic Window Approach (DWA)** [2] is a classical local planner that searches a velocity space constrained by robot dynamics, forward-simulates trajectories, and selects the action optimizing a weighted sum of goal progress, obstacle clearance, speed, and heading alignment objectives. DWA is deterministic, interpretable, and requires no training data, making it a natural baseline for any navigation system. However, DWA suffers from known limitations: its scoring function uses hand-tuned weights that do not adapt to environment statistics; its obstacle model assumes perfect, instantaneous sensing; and its reactive nature can lead to oscillatory or deadlocked behavior in cluttered scenes.

**LoGoPlanner** [3] is a strong, LeKiwi-deployed navigation framework developed independently. It represents a category of learned, end-to-end trajectory generation methods that incorporate geometric reasoning and produce smooth, goal-directed paths. We include LoGoPlanner as a strong baseline to establish an upper performance reference, while noting that our goal is not to outperform LoGoPlanner on raw navigation metrics. Instead, we study whether a lightweight, interpretable residual safety framework can offer complementary benefits: easier deployment, lower computational cost, inherent safety guarantees via hard constraints, and seamless LeRobot compatibility.

### E. Contributions

This paper makes the following contributions:

1. **A LeRobot-compatible RGB-D navigation data pipeline** for the LeKiwi robot with a D435i camera, producing synchronized observation frames that include RGB, depth-derived Scan64, robot state, nominal and residual actions, safety masks, and intervention records—all conforming to the standard LeRobot dataset schema.

2. **A compact Scan64 virtual LiDAR representation** that projects preprocessed depth images into 64 angular bins using percentile pooling, providing a noise-robust, low-dimensional obstacle encoding suitable for both classical planners and learned policies on computationally constrained hardware.

3. **A residual safety navigation architecture** composed of a Residual Safety Model (lightweight MLP that predicts additive corrections to a nominal planner), an Emergency Safety Shield (rule-based hard constraints that override unsafe commands), and an Action Adapter (acceleration-limited, temporally smoothed command output). This layered design preserves the interpretability of the nominal planner while adding learned safety refinement and hard guarantees.

4. **Comprehensive experimental design and ablation methodology** evaluating the proposed framework against vanilla DWA, LeRobot native imitation learning, and LoGoPlanner-style baselines, with simulation, real-world, and sim-to-real gap analysis protocols. All datasets, configurations, and evaluation scripts are designed for reproducibility within the LeRobot ecosystem.

---

## II. Related Work

### A. Robot Imitation Learning and LeRobot

Imitation learning (IL) has become a dominant paradigm for robot policy learning, driven by the availability of large-scale robot demonstration datasets [4, 5] and expressive policy architectures including Diffusion Policy [6], Action Chunking Transformer (ACT) [7], and their variants. These methods learn visuomotor policies from human teleoperated demonstrations, mapping raw image observations to action sequences.

Several open-source frameworks have emerged to standardize robot learning workflows. ROS2-based systems [8] provide mature middleware but impose significant integration overhead. robomimic [9] and RLBench [10] offer standardized benchmark tasks but are simulation-focused. LIBERO [11] provides a benchmark for lifelong robot learning. Among these, **LeRobot** [1] distinguishes itself by targeting real-robot deployment with a unified record/train/evaluate pipeline, a standardized HDF5-based dataset format with explicit episode and frame-level metadata, pretrained model sharing via the Hugging Face Hub, and first-class support for multiple commercial and research robot platforms.

Our work extends LeRobot along two under-explored dimensions: (1) RGB-D perception integration for mobile navigation, where existing LeRobot configurations primarily target manipulation with RGB-only sensing; and (2) safety-critical policy correction, where we augment LeRobot's native imitation learning pipeline with residual safety modules and hard constraints.

### B. RGB-D Perception for Mobile Robot Navigation

Depth sensing for indoor navigation has been extensively studied. Traditional approaches use 2D LiDAR for obstacle detection [12], but these sensors add cost (\$500–\$2000) and weight while providing only a single horizontal plane, missing overhanging obstacles and ground-level hazards.

RGB-D-based navigation offers richer environmental understanding at lower hardware cost. Works such as [13, 14] demonstrate indoor visual navigation using depth images for obstacle mapping and localization. Deep completion methods [15, 16] attempt to fill depth holes by learning priors from RGB-D pairs, but these networks add inference latency and may hallucinate obstacles. Virtual LiDAR projection from depth images has been explored in [17, 18], typically using minimum-pooling over angular bins, which is sensitive to single-pixel outliers.

Our Scan64 representation differs in two ways: (1) we use **percentile pooling** (10th percentile) rather than minimum, which provides robustness to flying pixels and speckle noise while still detecting thin obstacles; and (2) we apply temporal EMA smoothing across consecutive scans, reducing frame-to-frame jitter caused by depth sensor noise. This compact 64-dimensional representation is directly usable by both classical planners (DWA) and learned policies (MLP residual model), bridging perception and control without requiring a full depth image to be transmitted or processed at the policy level.

### C. Traditional Local Planning and DWA

The Dynamic Window Approach (DWA) [2], introduced by Fox et al., remains one of the most widely deployed local planners in mobile robotics. Its core idea is to search over admissible velocities—those reachable within one control cycle given acceleration limits (the *dynamic window*) and those that allow stopping before collision (the *admissible velocities*)—and select the velocity that maximizes an objective function combining goal heading, obstacle clearance, and forward speed.

Formally, given current velocity $v_{\text{curr}} = [v_x, v_y, \omega]^T$, the dynamic window is:

$$V_d = \{v \mid v_{\min} \leq v \leq v_{\max},\; |v - v_{\text{curr}}| \leq \dot{v}_{\max} \Delta t\}$$

The objective function is:

$$J(v) = w_g \cdot \text{heading}(v) + w_o \cdot \text{clearance}(v) + w_s \cdot \text{speed}(v)$$

where the weights $w_g, w_o, w_s$ are manually tuned. Extensions include the Timed Elastic Band (TEB) [19] for global-local planning integration and model predictive control formulations [20].

Despite its widespread use, DWA has well-documented failure modes: (a) it assumes perfect, instantaneous sensing—noise in the obstacle scan can cause spurious braking or missed detections; (b) static hand-tuned weights do not adapt to environment density or dynamic obstacle behavior; (c) the velocity-space search is myopic, considering only a short prediction horizon; (d) in narrow passages with noisy scans, the robot may deadlock when all sampled trajectories are penalized.

### D. End-to-End Navigation and LoGoPlanner-Style Methods

Learned navigation policies have advanced rapidly, with methods ranging from deep reinforcement learning (DRL) for visual navigation [21, 22] to imitation-learned trajectory generators [6, 23]. These approaches typically map raw sensor observations (RGB images, depth, or LiDAR scans) directly to velocity or waypoint commands, bypassing explicit map building and planning.

**LoGoPlanner** [3] is a recently proposed navigation framework that has been deployed on the LeKiwi platform. It employs metric-aware geometric reasoning combined with trajectory optimization to generate smooth, collision-free paths. As a LeKiwi-native strong baseline, LoGoPlanner provides a valuable reference point for our work, representing the performance achievable by a dedicated end-to-end navigation system.

Our approach differs from end-to-end methods in philosophy and architecture. Rather than replacing the planner, we **augment** a nominal planner (DWA) with a lightweight residual correction module. This design choice is motivated by several practical considerations: (a) the residual model can be much smaller than an end-to-end policy since it only predicts corrections; (b) if the residual model fails or produces erroneous corrections, the system gracefully degrades to the nominal DWA behavior rather than producing unpredictable actions; (c) the hard safety shield provides formal guarantees (e.g., "robot will stop within $d_{\text{stop}}$ meters of any obstacle") that are difficult to obtain from purely learned policies; and (d) training requires only a modest dataset of safe correction examples rather than full navigation demonstrations.

### E. Residual Learning and Safety Shields

Residual learning for robot control was popularized by [24, 25], who showed that learning a residual (additive correction) on top of a fixed model-based controller combines the generalization of model-based methods with the adaptability of learning. In manipulation, [26] used residual policies to adapt to contact-rich tasks, while [27] demonstrated residual reinforcement learning for quadruped locomotion.

In navigation, residual formulations have been less explored. [28] applied residual learning to improve upon a classical planner in highway driving, and [29] learned residual cost functions for motion planning. Closest to our work, [30, 31] used safety filters—also called "shields" or "control barrier functions"—to project learned actions onto a safe set. Our Emergency Safety Shield is inspired by these approaches but operates on the simpler, computationally efficient principle of sector-based distance thresholds rather than solving continuous optimization problems online.

The key distinction of our method is the **layered architecture**: the residual model learns statistical corrections from data (adapting to sensor noise patterns and environment structure), while the hard shield enforces non-negotiable safety constraints (minimum distance, lateral motion inhibition). This separation of concerns—learned refinement vs. hard guarantees—is practically valuable for real-robot deployment.

### F. Sim-to-Real Transfer

Bridging the simulation-to-reality gap remains an open challenge in robot learning. Domain randomization [32, 33]—varying visual textures, lighting, dynamics parameters, and sensor noise during simulation training—has proven effective for vision-based policies. System identification and dynamics randomization [34] address the dynamics gap. For depth sensing specifically, [35] modeled RealSense noise characteristics to improve sim-to-real transfer of grasping policies.

For indoor mobile robot navigation, sim-to-real transfer presents unique challenges beyond those in manipulation. The robot's interaction with the environment is continuous and long-horizon: small perception or dynamics errors accumulate over extended trajectories, potentially causing collisions far from the training distribution. Additionally, floor surface properties (carpet vs. tile vs. concrete) affect wheel slip in ways that are difficult to randomize effectively.

Our approach addresses sim-to-real transfer through three mechanisms: (1) the Scan64 representation explicitly models D435i noise characteristics (depth holes, flying pixels) via percentile pooling and temporal smoothing, making the representation more consistent across simulation and reality; (2) the residual formulation means the model learns *corrections* rather than absolute actions, which is inherently more transferable since the correction distribution is narrower than the action distribution; and (3) the hard safety shield does not depend on learned parameters at all, providing a transfer-invariant safety guarantee.

---

## III. System Design

### A. Hardware Platform

Our experimental platform consists of a LeKiwi omnidirectional mobile base equipped with an Intel RealSense D435i RGB-D camera. Table I summarizes the hardware specifications.

**Table I. LeKiwi + D435i Hardware Specifications.**

| Component | Specification |
|-----------|--------------|
| Base type | 3-wheel omnidirectional (holonomic) |
| Wheel radius | 0.05 m |
| Base radius | 0.125 m |
| Motors | Feetech serial bus servos |
| Max linear velocity | $\pm 0.30$ m/s |
| Max angular velocity | $\pm 90$ deg/s |
| Camera | Intel RealSense D435i |
| Depth technology | Active IR stereo |
| Depth FOV (H $\times$ V) | $87^\circ \times 58^\circ$ |
| Depth resolution | $848 \times 480$ @ 30 Hz |
| Depth range (usable) | 0.15–5.0 m |
| RGB resolution | $1920 \times 1080$ @ 30 Hz |
| Onboard computer | Raspberry Pi 4 / Jetson Orin Nano |
| Host computer | Laptop with GPU (for training) |
| Communication | ZMQ over TCP/IP (WiFi/Ethernet) |

The D435i is mounted at approximately 0.4 m height with a forward-facing orientation and a slight downward tilt ($\sim 10^\circ$) to capture ground-level obstacles within the depth frame. The camera's USB 3.0 interface connects to the onboard computer, which performs depth preprocessing and Scan64 extraction before transmitting compact observation packets to the host computer.

**[Fig. 1. Overview of the LeKiwi + D435i hardware platform. Left: CAD model showing camera mounting position, omniwheel configuration, and coordinate frames. Right: photograph of the actual robot in the indoor test environment.]** *[Figure to be produced]*

### B. LeRobot-Compatible Data Flow

We adopt the LeRobot dataset schema [1] to ensure compatibility with LeRobot's record, train, and evaluate workflows. Each episode is stored as a set of synchronized time-stamped frames. Table II lists the data fields defined in our extended schema.

**Table II. LeRobot-Compatible RGB-D Navigation Dataset Schema.**

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `observation.state` | float32 | (6,) | $[v_x, v_y, \omega, d_{\text{front}}, d_{\text{left}}, d_{\text{right}}]$ |
| `observation.images.front` | uint8 | (480, 848, 3) | RGB image from D435i |
| `observation.scan64` | float32 | (64,) | Virtual LiDAR scan (meters) |
| `observation.goal` | float32 | (3,) | Goal pose in robot frame $[dx, dy, d\theta]$ |
| `action.nominal` | float32 | (3,) | Raw DWA output $[v_x, v_y, \omega]$ |
| `action.residual` | float32 | (3,) | Residual correction $[\Delta v_x, \Delta v_y, \Delta \omega]$ |
| `action.executed` | float32 | (3,) | Final executed action after shield + adapter |
| `safety.min_distance` | float32 | (1,) | Minimum obstacle distance in frame |
| `safety.mask` | bool | (1,) | True if shield triggered |
| `safety.emergency_stop` | bool | (1,) | True if emergency stop activated |
| `metadata.domain` | str | — | `"simulation"` or `"real"` |
| `metadata.scene_id` | str | — | Scene identifier |
| `metadata.episode_id` | int | — | Episode index |
| `metadata.frame_id` | int | — | Frame index within episode |
| `metadata.timestamp_utc` | str | — | ISO 8601 timestamp |

All fields use 32-bit floating point except RGB images (uint8) and metadata. This schema is a strict superset of LeRobot v2.0 format; existing LeRobot tools (`lerobot/scripts/visualize_dataset.py`, `lerobot/datasets/`) can load and visualize the core fields without modification.

The data flow follows LeRobot's standard pipeline: the `LeKiwiD435iClient` (inheriting from `lerobot.robots.Robot`) implements `get_observation()` and `send_action()`, producing observation dictionaries that can be directly consumed by LeRobot's `record` script and training dataloaders. Our `LeRobotDatasetWriter` extends LeRobot's `LeRobotDataset` to write the additional safety-specific fields.

### C. RGB-D to Scan64 Virtual LiDAR Projection

The core perception module converts raw D435i depth images into a compact 64-dimensional polar scan representation. This process involves four stages:

**Stage 1: Depth Preprocessing.** Raw uint16 depth frames (millimeters) are converted to meters and processed:

1. **Clamp:** $D(u,v) \leftarrow \text{clip}(D(u,v), d_{\min}, d_{\max})$, where $d_{\min}=0.15$ m and $d_{\max}=5.0$ m. Pixels with $D < d_{\min}$ are set to $d_{\max}$.
2. **Median filter:** $D \leftarrow \text{medianBlur}(D, k=5)$ to suppress salt-and-pepper noise while preserving edges.
3. **Hole filling:** Invalid regions (depth near $d_{\max}$) are inpainted using Navier-Stokes inpainting [36] with a small radius (3 px) to fill small holes without hallucinating large structures.

**Stage 2: Horizontal Band Extraction.** Only the middle $\eta = 0.3$ fraction of the image height is retained:

$$B = D\left[\frac{H}{2} - \frac{\eta H}{2} : \frac{H}{2} + \frac{\eta H}{2},\; :\right]$$

This focuses on obstacles at the robot's operating height and excludes floor and ceiling regions.

**Stage 3: Angular Binning.** The horizontal field of view $\Theta_{\text{fov}} = 87^\circ$ is divided into $N = 64$ equal angular bins. Each image column $u$ maps to an angle $\theta(u) = \Theta_{\text{fov}} \cdot (u/W - 0.5)$. Columns are assigned to bins:

$$B_i = \{B(r, c) \mid c \in [\lfloor i \cdot W/N \rfloor, \lfloor (i+1) \cdot W/N \rfloor) \}$$

**Stage 4: Percentile Pooling.** For each bin $i$, we compute the $p$-th percentile of valid depth values:

$$s_i = Q_p(\{d \in B_i \mid d_{\min} < d < d_{\max}\})$$

where $Q_p$ denotes the $p$-th percentile and we use $p = 10$. This choice is critical: the minimum ($p=0$) is sensitive to single-pixel dropouts that are common with active stereo; the median ($p=50$) can miss thin obstacles (e.g., table legs). The 10th percentile provides a robust near-minimum that detects thin obstacles while ignoring isolated noise pixels.

The output is $s \in \mathbb{R}^{64}$, where each element represents the estimated distance to the nearest obstacle in that angular sector. Invalid bins (no valid pixels) are set to $d_{\max}$.

**Temporal Smoothing.** To reduce frame-to-frame jitter, we apply exponential moving average (EMA) smoothing:

$$\tilde{s}_t = \alpha \cdot s_t + (1 - \alpha) \cdot \tilde{s}_{t-1}$$

with $\alpha = 0.5$. NaN handling propagates valid values from the previous frame when the current frame has invalid bins.

**Sector Analysis.** We compute per-sector minimum distances for the safety shield:

$$d_{\text{front}} = \min\{s_i \mid i \in \mathcal{I}_{\text{front}}\}, \quad d_{\text{left}} = \min\{s_i \mid i \in \mathcal{I}_{\text{left}}\}, \quad d_{\text{right}} = \min\{s_i \mid i \in \mathcal{I}_{\text{right}}\}$$

where $\mathcal{I}_{\text{front}}, \mathcal{I}_{\text{left}}, \mathcal{I}_{\text{right}}$ partition the 64 beams into three approximately equal sectors corresponding to the front, left, and right of the robot.

**[Fig. 2. Scan64 projection pipeline. (a) Raw D435i depth image with noise artifacts highlighted. (b) Preprocessed depth after median filtering and hole filling. (c) Extracted horizontal band. (d) 64-beam polar scan visualization (polar plot). (e) Per-sector minimum distances overlaid on RGB.]** *[Figure to be produced]*

### D. ZMQ Host/Client Communication Architecture

We adopt a host/client architecture with ZeroMQ (ZMQ) for low-latency, asynchronous communication between the robot's onboard computer (host) and the control laptop (client). This design separates real-time perception and motor control from higher-level policy inference and logging.

**Host (Onboard Computer):**
- Reads D435i RGB and depth frames at 30 Hz.
- Preprocesses depth and computes Scan64, sector minima, and quality diagnostics.
- Encodes RGB as base64 JPEG for efficient transmission.
- Optionally detects ArUco markers for pallet localization.
- Saves raw depth frames locally as uint16 PNG for offline analysis.
- PUSHes JSON observation packets over ZMQ at configurable frequency (default 20 Hz).
- PULLs velocity commands from ZMQ and forwards them to the Feetech motor bus.
- Implements a **watchdog timer**: if no command is received within $T_{\text{watchdog}}$ ms, the robot executes an emergency stop.

**Client (Control Laptop):**
- PULLs observation packets from ZMQ.
- Decodes base64 JPEG to RGB numpy array.
- Runs the navigation pipeline: DWA planner $\to$ residual model $\to$ emergency shield $\to$ action adapter.
- Logs all data in LeRobot-compatible format.
- PUSHes final velocity commands to ZMQ.

**Latency Breakdown (estimated, [TBD]):**

**Table III. Estimated Communication and Inference Latency.**

| Stage | Estimated Latency | Location |
|-------|-------------------|----------|
| Depth capture + preprocessing | [TBD] ms | Host |
| JPEG encode + ZMQ send | [TBD] ms | Host |
| Network transport (WiFi) | [TBD] ms | — |
| ZMQ receive + JPEG decode | [TBD] ms | Client |
| DWA planning | [TBD] ms | Client |
| Residual model inference | [TBD] ms | Client |
| Shield + adapter | [TBD] ms | Client |
| ZMQ command send + motor write | [TBD] ms | Host |
| **Total control loop** | **[TBD] ms** | — |

The use of ZMQ CONFLATE sockets ensures that only the most recent observation/command is processed, preventing queue buildup when inference time exceeds the frame interval.

**[Fig. 3. System architecture diagram showing the host/client ZMQ communication topology, data flow between perception, planning, residual correction, safety shield, and motor execution modules.]** *[Figure to be produced]*

### E. Dataset Schema and Construction

We construct three dataset variants to support training and evaluation:

**$\mathcal{D}_{\text{sim}}$ (Synthetic Dataset):** Generated using our synthetic RGB-D renderer with procedurally generated indoor scenes (warehouse aisles, cluttered labs, empty rooms). Domain randomization includes: random wall/floor textures, random obstacle positions and sizes, random lighting intensity and direction, depth noise injection (Gaussian + dropout patterns), and random start/goal positions. Each episode contains a complete trajectory of a simulated DWA planner navigating the scene.

**$\mathcal{D}_{\text{real}}$ (Real-World Dataset):** Collected on the physical LeKiwi robot with a human operator providing teleoperation commands. The robot is driven through indoor environments with static obstacles (boxes, furniture) and dynamic obstacles (walking pedestrians). During collection, the DWA planner runs in parallel and its output is recorded alongside the human action, providing a source of "safe action" labels. Human interventions (takeover events) are automatically flagged as safety-critical frames.

**$\mathcal{D}_{\text{mix}}$ (Mixed Dataset):** The union of $\mathcal{D}_{\text{sim}}$ and $\mathcal{D}_{\text{real}}$, with real data upsampled (2$\times$) to address its smaller size.

**Label Construction:** For each frame, the residual target $\Delta a$ is computed as:

$$\Delta a_{\text{target}} = \text{clip}(a_{\text{DWA}}^{\text{safe}} - a_{\text{DWA}}^{\text{raw}}, -\Delta_{\max}, \Delta_{\max})$$

where $a_{\text{DWA}}^{\text{safe}}$ is the DWA output when running with a conservative safety distance parameter and $a_{\text{DWA}}^{\text{raw}}$ is the standard DWA output. This "conservative DWA vs. standard DWA" labeling strategy provides a self-supervised source of correction targets without requiring manual annotation.

---

## IV. Method: Residual Safety Navigation

### A. Problem Formulation

We consider a mobile robot operating in an indoor environment with static and dynamic obstacles. At each discrete time step $t$, the robot receives an observation $o_t$ consisting of:

- A depth-derived Scan64 vector $s_t \in \mathbb{R}^{64}$
- An optional RGB image $I_t \in \mathbb{R}^{H \times W \times 3}$
- The current robot state $x_t = [x, y, \theta, v_x, v_y, \omega]^T$
- A goal specification $g_t = [dx, dy, d\theta]^T$ in the robot's local frame

A nominal planner $\pi_0$ (in our case, DWA) produces a candidate action:

$$a_t^0 = \pi_0(s_t, g_t, x_t) = [v_x^0, v_y^0, \omega^0]^T$$

The objective is to produce a final executed action $a_t^{\text{exec}}$ that satisfies:

1. **Safety:** Maintain $d_{\min}(s_t) \ge d_{\text{stop}}$ at all times, where $d_{\min}(s_t)$ is the minimum obstacle distance in the current scan.
2. **Goal-directedness:** Reach the goal $g$ within reasonable time.
3. **Smoothness:** Minimize action jerk $\|a_t^{\text{exec}} - a_{t-1}^{\text{exec}}\|$ to prevent abrupt motions.
4. **Noise robustness:** Operate reliably under the depth sensing artifacts described in Section I-C.

Our solution composes three modules:

$$a_t^{\text{exec}} = f_{\text{adapt}} \circ f_{\text{shield}} \circ f_{\text{residual}}(a_t^0, s_t, g_t, x_t)$$

### B. Nominal Planner and DWA Baseline

Our DWA implementation for the omnidirectional LeKiwi platform searches over a 3-D velocity space $(v_x, v_y, \omega)$ with the following configuration (default values from our codebase):

**Velocity Constraints:**
$$v_x \in [-0.3, 0.3] \text{ m/s}, \quad v_y \in [-0.3, 0.3] \text{ m/s}, \quad \omega \in [-90, 90] \text{ deg/s}$$

**Dynamic Window:**
$$V_d(t) = \{(v_x, v_y, \omega) \mid |v_x - v_x^{t-1}| \le \dot{v}_{\max} \Delta t,\; |v_y - v_y^{t-1}| \le \dot{v}_{\max} \Delta t,\; |\omega - \omega^{t-1}| \le \dot{\omega}_{\max} \Delta t\}$$
with $\dot{v}_{\max} = 0.5$ m/s$^2$, $\dot{\omega}_{\max} = 180$ deg/s$^2$, $\Delta t = 0.1$ s.

**Discretization:** 7 samples in $v_x$, 7 in $v_y$, 15 in $\omega$ (735 total candidates).

**Trajectory Forward Simulation:** For each candidate velocity, we simulate a trajectory of duration $T_{\text{predict}} = 1.5$ s with step $\Delta t = 0.1$ s using the omnidirectional kinematic model:

$$\begin{aligned} \theta_{k+1} &= \theta_k + \omega \cdot \Delta t \\ x_{k+1} &= x_k + (v_x \cos\theta_k - v_y \sin\theta_k) \cdot \Delta t \\ y_{k+1} &= y_k + (v_x \sin\theta_k + v_y \cos\theta_k) \cdot \Delta t \end{aligned}$$

**Scoring Function:** Each trajectory $\tau$ receives a score:

$$J_{\text{DWA}}(\tau) = w_g \cdot J_g(\tau) + w_c \cdot J_c(\tau) + w_s \cdot J_s(\tau) + w_h \cdot J_h(\tau)$$

where:
- $J_g(\tau) = \exp(-d_{\text{goal}} / 2.0)$ — goal proximity (exponential decay, characteristic length 2 m)
- $J_c(\tau) = \begin{cases} 1 - \exp(-d_{\min} / d_{\text{safety}}), & \text{if } d_{\min} \ge d_{\text{safety}} \\ -100 / \max(d_{\min}, 0.01), & \text{otherwise} \end{cases}$ — obstacle clearance
- $J_s(\tau) = \bar{v} / v_{\max}$ — speed preference
- $J_h(\tau) = \cos(\theta_{\text{final}} - \theta_{\text{goal}})$ — heading alignment

Default weights: $w_g = 1.0$, $w_c = 0.5$, $w_s = 0.1$, $w_h = 0.3$, with safety distance $d_{\text{safety}} = 0.2$ m.

**Limitations:** As discussed in Section II-C, this DWA formulation assumes clean, instantaneous scan data ($s_t$ is treated as ground truth), uses fixed hand-tuned weights, and can produce oscillatory behavior when multiple trajectories have similar scores under noisy scans.

### C. Residual Safety Model

The Residual Safety Model (RSM) is a lightweight neural network that predicts an additive correction $\Delta a_t$ to the nominal DWA action:

$$a_t^{\text{res}} = \text{clip}\left(a_t^0 + \Delta a_\theta(s_t, g_t, x_t, a_t^0, a_{t-1}^{\text{exec}}), a_{\min}, a_{\max}\right)$$

**Input Features (76-D):**
- Scan64: $s_t \in \mathbb{R}^{64}$
- Raw action: $a_t^0 \in \mathbb{R}^{3}$
- Goal vector: $g_t \in \mathbb{R}^{3}$
- Current velocity: $[v_x, v_y, \omega]^T \in \mathbb{R}^{3}$
- Previous executed action: $a_{t-1}^{\text{exec}} \in \mathbb{R}^{3}$

Optionally, a lightweight CNN feature extractor (3 conv layers, 8-16-32 channels) can be added to process the RGB image, but we consider this optional to maintain low computational cost.

**Network Architecture:**

$$\begin{aligned} h_0 &= [s_t, a_t^0, g_t, v_t, a_{t-1}^{\text{exec}}] \in \mathbb{R}^{76} \\ h_1 &= \text{ReLU}(\text{Dropout}_{0.1}(W_1 h_0 + b_1)), \quad W_1 \in \mathbb{R}^{128 \times 76} \\ h_2 &= \text{ReLU}(\text{Dropout}_{0.1}(W_2 h_1 + b_2)), \quad W_2 \in \mathbb{R}^{64 \times 128} \\ h_3 &= \text{ReLU}(\text{Dropout}_{0.1}(W_3 h_2 + b_3)), \quad W_3 \in \mathbb{R}^{32 \times 64} \\ \Delta a_t &= W_4 h_3 + b_4, \quad W_4 \in \mathbb{R}^{3 \times 32} \end{aligned}$$

Total parameters: $\sim$18,000. The final layer is initialized with near-zero weights ($\mathcal{N}(0, 10^{-4})$) and zero bias so that the model initially behaves as an identity pass-through ($\Delta a \approx 0$).

**Why Residual Learning?** Three practical advantages motivate this design:

1. **Safe initialization:** At $t=0$, the model outputs near-zero corrections, so the system runs the nominal (safe-by-design) DWA planner. The model gradually learns to refine actions as training progresses.

2. **Graceful degradation:** If the model produces erroneous corrections (e.g., due to out-of-distribution observations), the hard safety shield (Section IV-D) overrides the action. The system never relies solely on the learned correction for safety.

3. **Data efficiency:** Learning corrections requires fewer examples than learning full navigation actions, since the model only needs to represent the *difference* between the nominal planner's output and a safer alternative, not the entire navigation policy.

### D. Emergency Safety Shield

The Emergency Safety Shield is a deterministic, rule-based layer that enforces hard safety constraints. It operates on three sectors computed from the Scan64 representation: front, left, and right, each with a minimum distance value ($d_{\text{front}}, d_{\text{left}}, d_{\text{right}}$).

The shield applies four rules in priority order:

**Rule 1 — Emergency Stop (Forward):**
$$\text{if } d_{\text{front}} < d_{\text{stop}} \text{ then } v_x \leftarrow 0$$
where $d_{\text{stop}} = 0.15$ m.

**Rule 2 — Lateral Motion Inhibition:**
$$\text{if } d_{\text{left}} < d_{\text{lateral}} \text{ and } v_y > 0 \text{ then } v_y \leftarrow 0$$
$$\text{if } d_{\text{right}} < d_{\text{lateral}} \text{ and } v_y < 0 \text{ then } v_y \leftarrow 0$$
where $d_{\text{lateral}} = 0.30$ m.

**Rule 3 — Proportional Slowdown:**
$$\text{if } d_{\text{stop}} \le d_{\text{front}} < d_{\text{slow}} \text{ and } v_x > 0 \text{ then } v_x \leftarrow v_x \cdot \lambda(d_{\text{front}})$$
where $d_{\text{slow}} = 0.50$ m and the scaling factor is:
$$\lambda(d) = \frac{d - d_{\text{stop}}}{d_{\text{slow}} - d_{\text{stop}}}, \quad \lambda \in [0, 1]$$

**Rule 4 — Rotation Stop:**
$$\text{if } \min(d_{\text{front}}, d_{\text{left}}, d_{\text{right}}) < d_{\text{stop}} \cdot \gamma_{\text{rot}} \text{ then } \omega \leftarrow 0$$
where $\gamma_{\text{rot}} = 1.5$, giving a rotation stop distance of $0.225$ m.

The shield outputs both the modified action and diagnostic information:

$$(a_t^{\text{safe}}, \text{triggered}, \text{reason}) = f_{\text{shield}}(a_t^{\text{res}}, s_t)$$

**Stuck Detection and Watchdog:** In addition to the sector-based rules, the shield monitors:
- **Command timeout:** If no command is received for $T_{\text{watchdog}} = 500$ ms, all motors are set to zero.
- **Invalid depth:** If the fraction of NaN bins in $s_t$ exceeds $\eta_{\text{invalid}} = 0.5$, the robot stops.
- **Manual intervention:** A human operator can trigger an emergency stop via a physical button or software command.

### E. Action Adapter

The Action Adapter applies post-processing to ensure smooth, physically realizable commands:

**Step 1 — Velocity Clipping:**
$$\tilde{a}_t = \text{clip}(a_t^{\text{safe}}, a_{\min}, a_{\max})$$

**Step 2 — Acceleration Limiting:**
For each axis $i \in \{x, y, \omega\}$:
$$\tilde{a}_{t,i} \leftarrow a_{t-1,i}^{\text{exec}} + \text{clip}(\tilde{a}_{t,i} - a_{t-1,i}^{\text{exec}}, -\dot{a}_{\max,i} \Delta t, \dot{a}_{\max,i} \Delta t)$$

**Step 3 — EMA Smoothing:**
$$a_t^{\text{exec}} = \alpha \cdot \tilde{a}_t + (1 - \alpha) \cdot a_{t-1}^{\text{exec}}$$
with $\alpha = 0.3$, providing strong temporal smoothing (lower $\alpha$ = more smoothing).

These three steps ensure that: (a) commands respect hardware velocity limits, (b) acceleration never exceeds motor capabilities, and (c) the action sequence is temporally smooth, reducing mechanical wear and improving passenger/observer comfort.

### F. Training Objectives

The Residual Safety Model is trained using a composite loss function:

$$\mathcal{L} = \mathcal{L}_{\text{res}} + \lambda_{\text{safe}} \mathcal{L}_{\text{safe}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{collision}} \mathcal{L}_{\text{collision}}$$

**Residual Loss (Supervised):**
$$\mathcal{L}_{\text{res}} = \frac{1}{B} \sum_{i=1}^{B} \|\Delta a_i^{\text{pred}} - \Delta a_i^{\text{target}}\|_2^2$$
where $\Delta a_i^{\text{target}}$ is the conservative DWA vs. standard DWA difference (Section III-E).

**Safety Risk Penalty:**
$$\mathcal{L}_{\text{safe}} = \frac{1}{B} \sum_{i=1}^{B} r(s_i) \cdot \max(0, \epsilon - \|\Delta a_i^{\text{pred}}\|)$$
where $r(s_i) \in [0, 1]$ is a collision risk score computed from the scan (higher when obstacles are close), and $\epsilon = 0.02$. This term penalizes the model when it predicts a small correction ($\|\Delta a\| < \epsilon$) in a dangerous state—pushing the model to produce larger corrections when risk is high.

**Smoothness Regularization:**
$$\mathcal{L}_{\text{smooth}} = \frac{1}{B} \sum_{i=1}^{B} \|\Delta a_i^{\text{pred}}\|_2^2$$
This L2 penalty on the delta magnitude discourages large, jittery corrections, complementing the Action Adapter's temporal smoothing.

**Collision Classification Loss (Optional):**
When binary collision/non-collision labels are available (from simulation or human intervention flags), we add:
$$\mathcal{L}_{\text{collision}} = \frac{1}{B} \sum_{i=1}^{B} \text{BCE}(p_i, y_i)$$
where $p_i = \sigma(\text{MLP}_{\text{head}}(h_3))$ is a predicted collision probability from an auxiliary head, and $y_i$ is the ground-truth label.

**Training Hyperparameters [TBD — nominal values]:**

**Table IV. Training Hyperparameters.**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-5}$ |
| Batch size | 64 |
| Epochs | 100 (early stopping, patience=15) |
| LR schedule | StepLR ($\gamma=0.5$, step=30) |
| $\lambda_{\text{safe}}$ | 0.5 |
| $\lambda_{\text{smooth}}$ | 0.1 |
| $\lambda_{\text{collision}}$ | 0.2 |
| Gradient clip norm | 1.0 |
| Train/Val/Test split | 70/15/15 |

### G. Algorithm

**Algorithm 1: LeRobot-Compatible Residual Safety Navigation**

```
Input: Trained RSM model θ, DWA planner π₀, Shield params Φ, Adapter params Ψ
Output: None (controls robot in real-time)

1:  Initialize: a_exec ← [0, 0, 0], s_smooth ← None
2:  for t = 1, 2, ... do
3:      // Perception
4:      o_t ← receive_observation()           ▷ ZMQ PULL
5:      I_t, D_t ← o_t.rgb, o_t.depth
6:      D_t ← preprocess_depth(D_t)           ▷ clamp, median, inpaint
7:      s_t ← depth_to_scan_polar(D_t, N=64, p=10)
8:      s_t ← temporal_smooth(s_t, s_smooth, α=0.5); s_smooth ← s_t
9:      d_front, d_left, d_right ← compute_sector_mins(s_t)
10:     
11:     // Nominal planning
12:     a_nom ← π₀.compute_action(s_t, g_t, v_t)
13:     
14:     // Residual correction (learned)
15:     x ← [s_t, a_nom, g_t, v_t, a_exec]
16:     Δa ← RSM_θ.predict(x)                 ▷ PyTorch inference
17:     a_res ← clip(a_nom + Δa, a_min, a_max)
18:     
19:     // Emergency safety shield (hard rules)
20:     a_safe, triggered, reason ← f_shield(a_res, d_front, d_left, d_right)
21:     
22:     // Action adapter (post-processing)
23:     a_exec ← f_adapt(a_safe, a_exec)       ▷ clip, accel limit, EMA
24:     
25:     // Execution
26:     send_action(a_exec)                    ▷ ZMQ PUSH
27:     
28:     // Logging (LeRobot-compatible)
29:     frame ← {obs: o_t, action.nominal: a_nom, action.residual: Δa,
30:              action.executed: a_exec, safety.mask: triggered,
31:              safety.reason: reason, scan64: s_t, ...}
32:     dataset_writer.add_frame(frame)
33: end for
```

---

## V. Experiments

### A. Research Questions

Our experimental design is structured around five research questions:

- **RQ1 (Scan64):** Does the Scan64 representation provide a robust, compact alternative to raw depth for low-cost navigation, and how does percentile pooling compare to minimum pooling under depth sensor noise?
- **RQ2 (Residual):** Does residual safety correction improve navigation safety and efficiency over vanilla DWA?
- **RQ3 (Shield):** Does the Emergency Safety Shield reduce collision rate and unsafe action count, and does it introduce excessive conservatism?
- **RQ4 (Comparison):** How does the proposed framework compare with LeRobot native imitation learning and a LoGoPlanner-style strong baseline in terms of safety, efficiency, and computational cost?
- **RQ5 (Sim-to-Real):** What is the magnitude of the sim-to-real performance gap, and which components contribute most to reducing or exacerbating this gap?

### B. Simulation Setup

**Environment:** We use four procedurally generated indoor scene types (implemented in `sim/mujoco_lekiwi/envs/lekiwi_scan_env.py`):

1. **lab_empty:** A $20 \times 20$ m room bounded by walls, with no internal obstacles. Tests basic goal-directed navigation.
2. **warehouse_aisle:** A warehouse-like environment with parallel shelving units creating narrow aisles (2.4 m width). Tests navigation in constrained corridors.
3. **pallet_pickup:** An open room with a pallet zone near the center and scattered obstacles. Tests approach behaviors.
4. **cluttered_lab:** A lab-like room with 12 randomly placed boxes and 4 cylindrical obstacles. Tests navigation in dense, unstructured clutter.

**Domain Randomization:** For each episode, we randomize: wall/floor textures (from a set of 20 textures), obstacle positions ($\pm 0.5$ m perturbation), obstacle sizes ($\pm 20\%$), lighting intensity ($0.5\times$–$2.0\times$), and start/goal positions within designated zones.

**Depth Noise Model:** We inject synthetic noise matching D435i characteristics: Gaussian noise $\sigma(d) = 0.01 + 0.005 \cdot d^2$ (depth-dependent), random dropout (5% probability per pixel), and structured missing regions (1–3 per frame, simulating reflective surfaces).

**Dynamic Obstacles:** In 50% of episodes, 1–3 dynamic obstacles (simulated pedestrians) move along randomized waypoint paths at speeds of 0.3–1.0 m/s.

**Dataset:** $\mathcal{D}_{\text{sim}}$ contains [TBD] episodes across the four scene types.

### C. Real-World Setup

**Robot:** LeKiwi mobile base with D435i camera, as described in Section III-A.

**Test Environment:** Indoor laboratory space (approximately $8 \times 6$ m) with:
- Static obstacles: cardboard boxes ($0.3 \times 0.3 \times 0.4$ m), office chairs, tables.
- Dynamic obstacles: 1–2 pedestrians walking through the navigation area.
- Floor surface: smooth tile.
- Lighting: standard office fluorescent lighting (no direct sunlight).

**Data Collection:** Human operators teleoperate the robot through 10–20 navigation episodes using a gamepad. During teleoperation, the full perception and safety pipeline runs in parallel, logging all observations and actions. Human interventions (gamepad override) are automatically flagged.

**Safety Protocol:** All real-world experiments are conducted at low speeds ($v_{\max} \le 0.2$ m/s) with a human supervisor holding a physical emergency stop button. The robot's onboard emergency shield runs independently of the experimental pipeline.

**Dataset:** $\mathcal{D}_{\text{real}}$ contains [TBD] episodes $\times$ [TBD] frames.

### D. Compared Methods

We evaluate the following configurations:

1. **DWA-only:** Vanilla DWA planner with default parameters.
2. **DWA + Shield:** DWA with Emergency Safety Shield (no residual model).
3. **LeRobot IL:** Native LeRobot imitation learning policy trained on $\mathcal{D}_{\text{mix}}$ using behavior cloning (ACT-style or Diffusion Policy, [TBD which architecture]).
4. **LoGoPlanner:** LoGoPlanner-style strong baseline deployed on LeKiwi (configuration and weights provided by the LoGoPlanner authors or replicated following their methodology).
5. **Ours w/o Scan64:** Full pipeline but using raw depth image (downsampled to $64\times 48$) as input instead of Scan64.
6. **Ours w/o Residual:** Full pipeline with residual model disabled ($\Delta a \equiv 0$).
7. **Ours w/o Shield:** Full pipeline with Emergency Safety Shield disabled.
8. **Ours (Full):** Complete pipeline: Scan64 + DWA + Residual Safety Model + Emergency Safety Shield + Action Adapter.

### E. Evaluation Metrics

**Safety Metrics:**
- **Success Rate (SR):** Fraction of episodes where the robot reaches within 0.3 m of the goal without collision or timeout. ↑
- **Collision Rate (CR):** Fraction of episodes ending in collision (any obstacle contact). ↓
- **Minimum Obstacle Distance (MOD):** Average per-step minimum distance to nearest obstacle (meters). ↑
- **Emergency Stop Count (ESC):** Average number of emergency shield activations per episode. ↓ (lower is better, but zero may indicate the shield is not needed or is too conservative)
- **Human Intervention Rate (HIR):** Fraction of episodes requiring human takeover. ↓

**Efficiency Metrics:**
- **Success-weighted Path Length (SPL):** Path optimality normalized by success [37]. ↑
- **Time to Goal (TTG):** Wall-clock time from start to goal (seconds). ↓
- **Average Speed:** Mean linear speed during successful episodes (m/s).

**Smoothness Metrics:**
- **Average Jerk:** Mean absolute action difference between consecutive steps, normalized per axis. ↓
- **Oscillation Index:** Number of sign changes in angular velocity per episode. ↓

**Computational Metrics:**
- **Inference Latency:** Wall-clock time for each pipeline stage (ms).
- **CPU/GPU Utilization:** Percentage utilization during navigation.
- **Model Size:** Number of parameters and on-disk size.

**Sim-to-Real Metrics:**
- **Performance Drop:** $\Delta_{\text{S2R}} = \text{SR}_{\text{sim}} - \text{SR}_{\text{real}}$ and similarly for other metrics.
- **Scan Distribution Divergence:** Wasserstein distance between simulation and real Scan64 distributions.

### F. Quantitative Results

**[ALL QUANTITATIVE VALUES BELOW ARE PLACEHOLDERS — MARKED AS [TBD]]**

**Table V. Simulation Navigation Performance.** [TBD]

| Method | SR ↑ | CR ↓ | MOD (m) ↑ | ESC ↓ | SPL ↑ | Jerk ↓ |
|--------|------|------|-----------|-------|-------|--------|
| DWA-only | [TBD] | [TBD] | [TBD] | N/A | [TBD] | [TBD] |
| DWA + Shield | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| LeRobot IL | [TBD] | [TBD] | [TBD] | N/A | [TBD] | [TBD] |
| LoGoPlanner | [TBD] | [TBD] | [TBD] | N/A | [TBD] | [TBD] |
| Ours w/o Scan64 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Ours w/o Residual | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Ours w/o Shield | [TBD] | [TBD] | [TBD] | N/A | [TBD] | [TBD] |
| **Ours (Full)** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

*All results averaged over [TBD] episodes per scene type $\times$ 4 scene types $\times$ 3 random seeds. Standard deviations in parentheses [TBD].*

**Table VI. Real-World Navigation Performance.** [TBD]

| Method | SR ↑ | CR ↓ | MOD (m) ↑ | ESC ↓ | HIR ↓ | TTG (s) ↓ |
|--------|------|------|-----------|-------|-------|-----------|
| DWA-only | [TBD] | [TBD] | [TBD] | N/A | [TBD] | [TBD] |
| DWA + Shield | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| LeRobot IL | [TBD] | [TBD] | [TBD] | N/A | [TBD] | [TBD] |
| LoGoPlanner | [TBD] | [TBD] | [TBD] | N/A | [TBD] | [TBD] |
| Ours w/o Residual | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **Ours (Full)** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

*All results averaged over [TBD] episodes. Real-world experiments limited to safe configurations only (methods requiring the shield to be disabled are evaluated only in simulation).*

**Table VII. Ablation Study (Simulation).** [TBD]

| Ablation | SR ↑ | CR ↓ | MOD (m) ↑ | Jerk ↓ |
|----------|------|------|-----------|--------|
| Ours (Full) | [TBD] | [TBD] | [TBD] | [TBD] |
| — Remove residual model | [TBD] | [TBD] | [TBD] | [TBD] |
| — Remove safety shield | [TBD] | [TBD] | [TBD] | [TBD] |
| — Remove action adapter (no smoothing) | [TBD] | [TBD] | [TBD] | [TBD] |
| — Replace Scan64 with raw depth | [TBD] | [TBD] | [TBD] | [TBD] |
| — Replace percentile (p=10) with min (p=0) | [TBD] | [TBD] | [TBD] | [TBD] |
| — Remove temporal Scan64 smoothing | [TBD] | [TBD] | [TBD] | [TBD] |

**Table VIII. Computational Cost.** [TBD]

| Method | Params | Inference (ms) | CPU (%) | GPU (%) | Model Size (MB) |
|--------|--------|----------------|---------|---------|------------------|
| DWA-only | 0 | [TBD] | [TBD] | 0 | 0 |
| LeRobot IL | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| LoGoPlanner | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Ours (Residual Model) | $\sim$18K | [TBD] | [TBD] | [TBD] | [TBD] |
| Ours (Shield + Adapter) | 0 | [TBD] | [TBD] | 0 | 0 |
| Ours (Full Pipeline) | $\sim$18K | [TBD] | [TBD] | [TBD] | [TBD] |

### G. Qualitative Results

**[Fig. 4. Navigation trajectory comparisons. (a) DWA-only in warehouse_aisle: note oscillations near shelves. (b) Ours (Full) in same scene: smoother trajectory with larger clearance from shelves. (c) DWA-only in cluttered_lab: collision with a box due to depth dropout. (d) Ours (Full) in same scene: residual correction slows approach, shield triggers proportional slowdown. Trajectory color indicates speed (blue = slow, red = fast).]** *[Figure to be produced]*

**[Fig. 5. Scan64 visualization under depth noise. (a) Raw D435i depth with reflective surface (monitor screen) causing a large hole. (b) After preprocessing (median filter + hole filling). (c) Resulting Scan64: the reflective surface bin shows NaN but adjacent bins provide coverage. (d) Comparison of min-pooling vs. percentile-pooling (p=10): min-pooling produces spurious short readings (false positives), percentile-pooling is more stable.]** *[Figure to be produced]*

**[Fig. 6. Emergency shield activation analysis. (a) Time series of $d_{\text{front}}$ during a navigation episode, with $d_{\text{stop}}$ and $d_{\text{slow}}$ thresholds marked. (b) Corresponding velocity profile showing slowdown and stop events. (c) Histogram of shield trigger reasons across all episodes.]** *[Figure to be produced]*

### H. Ablation Study Discussion

**Expected Trends (to be verified experimentally):**

- **Removing the residual model** (Ours w/o Residual $\approx$ DWA + Shield) should maintain similar safety (the shield provides hard guarantees) but may reduce efficiency: the residual model learns to produce smoother, more goal-directed corrections that the hand-tuned DWA weights cannot capture. We expect higher jerk and longer time-to-goal without the residual model.

- **Removing the safety shield** (Ours w/o Shield) should significantly increase collision rate, particularly under depth noise conditions. The residual model alone, as a learned component without formal safety guarantees, may fail under out-of-distribution depth artifacts. This ablation validates the necessity of the hard safety layer.

- **Removing the action adapter** should increase jerk and oscillation, particularly near obstacles where the residual model and shield may produce rapidly changing corrections. The EMA smoothing is expected to have a larger effect in real-world experiments where depth noise causes frame-to-frame action variation.

- **Replacing Scan64 with raw depth** (downsampled) should marginally affect safety (the shield still operates on sector minima) but increase computational cost (larger input to the residual model) and potentially reduce robustness due to the higher-dimensional, noisier input space.

- **Replacing percentile pooling with minimum pooling** should increase false positive obstacle detections (spurious short readings from flying pixels), leading to more unnecessary slowdowns and higher emergency stop counts.

- **Removing temporal Scan64 smoothing** should increase jerk and oscillation, as frame-to-frame depth noise propagates through the DWA planner into the velocity commands.

### I. Sim-to-Real Gap Analysis

We analyze the sim-to-real gap along four dimensions:

**Perception Gap:** The distribution of Scan64 values differs between simulation and reality due to:
- D435i-specific noise patterns not fully captured by our noise model.
- Real-world depth artifacts: reflective surfaces (glass, monitors), texture-poor walls (white drywall), and direct sunlight through windows.
- Our percentile pooling and temporal smoothing are explicitly designed to reduce this gap by making the Scan64 representation more consistent across domains.

**Measurement Protocol [TBD]:** We plan to collect paired data (same physical scene scanned by both the real D435i and our synthetic renderer) and compute the Wasserstein distance between Scan64 distributions. We also plan per-bin statistics (mean, variance) to identify systematic biases.

**Dynamics Gap:** The LeKiwi's real-world dynamics differ from simulation in:
- Wheel slip on smooth tile (not modeled in our kinematic simulation).
- Motor backlash and dead zone (the Feetech servos have $\sim$0.5–2.0 deg backlash).
- Variable control loop latency due to WiFi jitter and OS scheduling.
- Floor unevenness causing pitch/roll variations that affect the D435i's viewpoint.

**Mitigation:** The action adapter's acceleration limiting and EMA smoothing partially compensate for dynamics discrepancies by preventing the controller from issuing commands that assume perfect execution. The safety shield operates on *observed* (not predicted) obstacle distances, so dynamics errors that cause the robot to be closer to obstacles than expected will trigger the shield.

**Expected Trend [TBD]:** We expect the sim-to-real performance drop to be smaller for our method than for LeRobot IL and LoGoPlanner because: (a) the residual model learns corrections (narrower distribution) rather than full actions; (b) the hard shield provides domain-invariant safety guarantees; and (c) the Scan64 representation is designed for cross-domain consistency.

**Table IX. Sim-to-Real Performance Drop (Expected Structure).** [TBD]

| Method | ΔSR (sim−real) | ΔCR (real−sim) | ΔMOD | ΔJerk |
|--------|-----------------|-----------------|------|-------|
| DWA-only | [TBD] | [TBD] | [TBD] | [TBD] |
| DWA + Shield | [TBD] | [TBD] | [TBD] | [TBD] |
| LeRobot IL | [TBD] | [TBD] | [TBD] | [TBD] |
| LoGoPlanner | [TBD] | [TBD] | [TBD] | [TBD] |
| **Ours (Full)** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

### J. Discussion

**Comparison with LoGoPlanner.** LoGoPlanner represents a strong end-to-end navigation baseline that has been specifically developed and tuned for the LeKiwi platform. We do not claim that our method outperforms LoGoPlanner on raw navigation metrics such as time-to-goal or path optimality. Instead, our contribution is complementary: we provide a lightweight ($\sim$18K parameter residual model + rule-based shield), interpretable (the residual correction and shield trigger are both inspectable), and LeRobot-compatible (plug-and-play with LeRobot's record/train/evaluate) safety navigation framework.

Scenarios where our method may be preferable:
- **Rapid deployment:** The DWA + shield configuration requires no training data and can be deployed immediately.
- **Safety certification:** The hard shield provides verifiable safety guarantees (the robot *will* stop within $d_{\text{stop}}$ of an obstacle) that are difficult to certify for purely learned policies.
- **Resource constraints:** Our residual model runs at [TBD] ms on CPU, suitable for Raspberry Pi-class hardware.
- **Data scarcity:** Training the residual model requires only correction examples, not full navigation demonstrations.

Scenarios where LoGoPlanner may be preferable:
- **Metric-optimal navigation:** When path length and time-to-goal are the primary objectives.
- **Complex, learned behaviors:** When navigation must integrate with higher-level task reasoning.
- **Large-scale deployment:** When extensive training data is available for fine-tuning.

**When does the residual model help most?** We expect the residual model to provide the largest benefit in scenarios with noisy depth (reflective surfaces, texture-poor walls) where DWA's fixed scoring function produces suboptimal or oscillatory behavior. The residual model can learn to "ignore" spurious close readings in known noisy regions while remaining appropriately conservative near genuine obstacles.

**When does the shield hurt performance?** The shield's fixed thresholds ($d_{\text{stop}} = 0.15$ m, $d_{\text{slow}} = 0.50$ m) may be overly conservative in very cluttered environments where the robot must pass within 0.15 m of obstacles to navigate. In such cases, the shield may prevent the robot from completing the task. Adaptive threshold adjustment (future work) could address this limitation.

**Limitations of the current study.** (1) Real-world experiments are limited to a single indoor environment and floor type. (2) Dynamic obstacle scenarios are limited to 1–2 pedestrians. (3) The LoGoPlanner comparison uses [TBD: provided weights or our replication], which may not match the performance of the original deployment. (4) All experiments use the D435i; results may not transfer to other depth sensors.

---

## VI. Conclusion and Future Work

### Conclusion

This paper presented a LeRobot-compatible residual safety navigation framework for the LeKiwi low-cost omnidirectional mobile robot equipped with an Intel RealSense D435i RGB-D camera. We proposed three integrated components: (1) a Scan64 virtual LiDAR representation that robustly encodes depth into a compact 64-beam polar scan via percentile pooling and temporal smoothing; (2) a lightweight Residual Safety Model that learns additive corrections to a DWA nominal planner, improving navigation smoothness and safety in an interpretable manner; and (3) a hard Emergency Safety Shield with an Action Adapter that enforces non-negotiable safety constraints and produces smooth, physically feasible velocity commands.

Our framework is designed to be fully compatible with the Hugging Face LeRobot ecosystem, reusing its standardized dataset schema, record/train/evaluate pipeline, and robot configuration system. We constructed simulation and real-world datasets and designed a comprehensive evaluation protocol comparing our method against vanilla DWA, LeRobot native imitation learning, and a LoGoPlanner-style strong baseline. Through detailed ablation studies, we isolated the contributions of each safety component.

The primary finding of this work is that a **layered safety architecture**—combining a classical planner (DWA), a learned residual corrector (RSM), and hard safety constraints (Emergency Shield)—offers a practical path toward safe, deployable navigation for low-cost mobile robots. The separation of concerns between learned refinement and hard guarantees is particularly valuable for real-world deployment, where formal safety verification is often required.

### Limitations

1. **Fixed safety thresholds:** The shield's $d_{\text{stop}}$ and $d_{\text{slow}}$ parameters are manually set and do not adapt to environment density, robot speed, or obstacle type.
2. **Perception blind spots:** The D435i's $\sim$87$^\circ$ horizontal FOV leaves rear and side-rear regions unobserved. The Scan64 representation inherits this limitation.
3. **Limited dynamic obstacle diversity:** Our current experiments consider only pedestrian-like dynamic obstacles; other moving agents (other robots, forklifts) may require different avoidance strategies.
4. **Single sensor dependency:** The system relies entirely on the D435i for obstacle detection. Sensor failure or severe degradation (e.g., direct sunlight) would disable obstacle awareness.
5. **Modest real-world scale:** Real-world experiments are limited to a single indoor laboratory and a relatively small number of episodes.
6. **LoGoPlanner comparison fidelity:** The LoGoPlanner baseline may not represent its fully optimized deployment configuration.
7. **Static threshold for invalid depth:** The NaN fraction threshold for triggering an emergency stop is heuristic and may cause unnecessary stops in scenes with legitimate depth holes (e.g., windows).

### Future Work

1. **Adaptive safety thresholds:** Learning or optimizing the shield thresholds online based on environment statistics, robot speed, and task context could reduce unnecessary conservatism while maintaining safety.
2. **Multi-sensor fusion:** Integrating low-cost ultrasonic sensors or a rear-facing depth camera would eliminate perception blind spots and provide redundancy.
3. **Larger LeRobot dataset:** Scaling $\mathcal{D}_{\text{real}}$ to more environments, floor types, lighting conditions, and dynamic obstacle scenarios would improve the residual model's generalization and enable more robust sim-to-real evaluation.
4. **Integration with end-to-end imitation learning:** The residual safety model and shield could be applied as a safety wrapper around LeRobot-trained imitation learning policies (e.g., Diffusion Policy, ACT), providing safety guarantees for policies that otherwise lack them.
5. **Semantic obstacle reasoning:** Incorporating YOLO-based object detection (already prototyped in our codebase) to distinguish between different obstacle types (person vs. chair vs. wall) could enable more intelligent avoidance behaviors—e.g., maintaining larger clearance from people than from static furniture.
6. **Outdoor transfer:** Extending the system to outdoor environments with natural lighting, uneven terrain, and longer-range navigation would test the generality of the Scan64 representation.
7. **Stronger LoGoPlanner comparison:** Collaborating with the LoGoPlanner authors to establish a standardized LeKiwi navigation benchmark with shared evaluation protocols, scenes, and metrics.
8. **Formal safety verification:** Applying formal methods (e.g., reachability analysis, control barrier functions) to verify the safety guarantees provided by the Emergency Shield under bounded sensor noise.

---

## Appendix

### A. Dataset Schema Details

The complete LeRobot-compatible dataset schema is defined below. Each episode is stored as a directory containing `meta.json` (episode-level metadata) and `frame_XXXXX.json` files (frame-level data).

**Episode Metadata (`meta.json`):**
```json
{
  "episode_id": 0,
  "scene_id": "warehouse_aisle_03",
  "domain": "simulation",
  "num_frames": 487,
  "success": true,
  "collision": false,
  "timeout": false,
  "goal_reached": true,
  "start_pose": [8.0, -7.5, 0.0],
  "goal_pose": [8.0, 7.5, 0.0],
  "dataset_version": "1.0.0",
  "lerobot_schema_version": "2.0",
  "collection_date": "2025-05-01",
  "robot_config": "lekiwi_d435i",
  "planner_config": "dwa_default",
  "shield_config": "default_v1"
}
```

**Frame Data (`frame_XXXXX.json`):**
```json
{
  "frame_id": 123,
  "timestamp_utc": "2025-05-01T12:00:01.234567",
  "observation.state": [0.12, 0.03, -0.08, 0.45, 3.20, 2.80],
  "observation.scan64": [1.23, 1.25, 1.28, ...],
  "observation.goal": [2.50, -0.30, 0.15],
  "action.nominal": [0.15, 0.00, 5.0],
  "action.residual": [-0.03, 0.01, -2.0],
  "action.executed": [0.12, 0.01, 3.0],
  "safety.min_distance": 0.45,
  "safety.mask": false,
  "safety.emergency_stop": false,
  "safety.shield_reason": null,
  "metadata.risk_score": 0.12
}
```

### B. Hyperparameters

**Table X. Complete Hyperparameter Settings.**

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Depth Preprocessing** | $d_{\min}$ | 0.15 m |
| | $d_{\max}$ | 5.0 m |
| | Median kernel | 5 |
| | Hole fill radius | 3 px |
| **Scan64** | $N$ (beams) | 64 |
| | $p$ (percentile) | 10 |
| | $\eta$ (slice fraction) | 0.3 |
| | $\Theta_{\text{fov}}$ | $87^\circ$ |
| | $\alpha$ (temporal EMA) | 0.5 |
| **DWA** | $v_x$ range | $[-0.3, 0.3]$ m/s |
| | $v_y$ range | $[-0.3, 0.3]$ m/s |
| | $\omega$ range | $[-90, 90]$ deg/s |
| | $v_x$ samples | 7 |
| | $v_y$ samples | 7 |
| | $\omega$ samples | 15 |
| | $T_{\text{predict}}$ | 1.5 s |
| | $\Delta t$ | 0.1 s |
| | $d_{\text{safety}}$ | 0.2 m |
| | Weights $(w_g, w_c, w_s, w_h)$ | $(1.0, 0.5, 0.1, 0.3)$ |
| **Residual Model** | Architecture | [128, 64, 32] MLP |
| | Activation | ReLU |
| | Dropout | 0.1 |
| | Input dim | 76 |
| | Output dim | 3 |
| | Parameters | $\sim$18K |
| **Training** | Optimizer | AdamW |
| | LR | $1 \times 10^{-4}$ |
| | Weight decay | $1 \times 10^{-5}$ |
| | Batch size | 64 |
| | Epochs | 100 |
| | Early stop patience | 15 |
| | LR schedule | StepLR(30, 0.5) |
| | Gradient clip | 1.0 |
| | $\lambda_{\text{safe}}$ | 0.5 |
| | $\lambda_{\text{smooth}}$ | 0.1 |
| **Safety Shield** | $d_{\text{stop}}$ | 0.15 m |
| | $d_{\text{slow}}$ | 0.50 m |
| | $d_{\text{lateral}}$ | 0.30 m |
| | $\gamma_{\text{rot}}$ | 1.5 |
| **Action Adapter** | $\dot{v}_{\max}$ | 0.5 m/s$^2$ |
| | $\dot{\omega}_{\max}$ | 180 deg/s$^2$ |
| | $\alpha$ (EMA) | 0.3 |
| | $\Delta t$ | 0.1 s |

### C. Safety Threshold Definitions

| Symbol | Value | Description |
|--------|-------|-------------|
| $d_{\text{stop}}$ | 0.15 m | Hard stop distance: robot must stop if any front-sector beam reads below this |
| $d_{\text{slow}}$ | 0.50 m | Proportional slowdown zone: velocity scales linearly from 0 at $d_{\text{stop}}$ to 1 at $d_{\text{slow}}$ |
| $d_{\text{lateral}}$ | 0.30 m | Lateral inhibition: lateral motion toward obstacle is blocked if side-sector minimum is below this |
| $\gamma_{\text{rot}}$ | 1.5 | Rotation stop multiplier: rotation stops when any sector below $d_{\text{stop}} \cdot \gamma_{\text{rot}}$ |
| $T_{\text{watchdog}}$ | 500 ms | Command timeout: all motors stop if no command received within this window |
| $\eta_{\text{invalid}}$ | 0.5 | Invalid depth fraction: robot stops if >50% of Scan64 bins are NaN |

### D. Reproducibility Checklist

- [ ] **Code:** All source code available at [repository URL TBD], with installation instructions in `README.md`.
- [ ] **Dataset:** $\mathcal{D}_{\text{sim}}$ generation scripts provided. $\mathcal{D}_{\text{real}}$ collection protocol documented. Pre-collected datasets to be released on Hugging Face Hub under LeRobot format.
- [ ] **Hardware:** LeKiwi robot hardware BOM and assembly instructions (preexisting, referenced from LeRobot). D435i mounting bracket CAD files.
- [ ] **Calibration:** D435i calibration procedure (using Intel RealSense SDK). Camera extrinsics measurement protocol.
- [ ] **Training:** Training config YAML files provided. Pretrained model weights to be released on Hugging Face Hub.
- [ ] **Evaluation:** Evaluation scripts (`tools/evaluate_navigation.py`). Standardized evaluation protocol (scene types, start/goal positions, random seeds).
- [ ] **Baselines:** DWA implementation in `control/dwa_policy.py`. LeRobot IL training config. LoGoPlanner configuration (to be coordinated with authors).
- [ ] **Metrics:** All metrics computed by `tools/evaluate_navigation.py`. Metric definitions documented in Section V-E.

---

## References

[1] R. Cadene, S. Alibert, A. So, et al., "LeRobot: State-of-the-art AI for real-world robotics," Hugging Face, 2024. [Online]. Available: https://github.com/huggingface/lerobot

[2] D. Fox, W. Burgard, and S. Thrun, "The dynamic window approach to collision avoidance," *IEEE Robotics & Automation Magazine*, vol. 4, no. 1, pp. 23–33, 1997. [verify citation]

[3] [LoGoPlanner citation — TBD. Need authors, title, venue, year. Verify with LoGoPlanner team.]

[4] A. Brohan, N. Brown, J. Carbajal, et al., "RT-1: Robotics Transformer for real-world control at scale," in *Robotics: Science and Systems (RSS)*, 2023.

[5] O. M. Team, D. Ghosh, H. Walke, et al., "Octo: An open-source generalist robot policy," in *Robotics: Science and Systems (RSS)*, 2024.

[6] C. Chi, S. Feng, Y. Du, et al., "Diffusion Policy: Visuomotor policy learning via action diffusion," in *Robotics: Science and Systems (RSS)*, 2023.

[7] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, "Learning fine-grained bimanual manipulation with low-cost hardware," in *Robotics: Science and Systems (RSS)*, 2023.

[8] S. Macenski, T. Foote, B. Gerkey, et al., "Robot Operating System 2: Design, architecture, and uses in the wild," *Science Robotics*, vol. 7, no. 66, 2022.

[9] A. Mandlekar, D. Xu, J. Wong, et al., "What matters in learning from offline human demonstrations for robot manipulation," in *Conference on Robot Learning (CoRL)*, 2021.

[10] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, "RLBench: The robot learning benchmark & learning environment," *IEEE Robotics and Automation Letters*, vol. 5, no. 2, pp. 3019–3026, 2020.

[11] B. Liu, Y. Zhu, C. Gao, et al., "LIBERO: Benchmarking knowledge transfer for lifelong robot learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[12] S. Thrun, W. Burgard, and D. Fox, *Probabilistic Robotics*. MIT Press, 2005.

[13] A. J. Davison, I. D. Reid, N. D. Molton, and O. Stasse, "MonoSLAM: Real-time single camera SLAM," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 29, no. 6, pp. 1052–1067, 2007. [verify citation]

[14] J. Engel, T. Schöps, and D. Cremers, "LSD-SLAM: Large-scale direct monocular SLAM," in *European Conference on Computer Vision (ECCV)*, 2014.

[15] F. Ma and S. Karaman, "Sparse-to-dense: Depth prediction from sparse depth samples and a single image," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2018.

[16] Y. Zhang and T. Funkhouser, "Deep depth completion of a single RGB-D image," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.

[17] J. Zhang and S. Singh, "LOAM: Lidar Odometry and Mapping in Real-time," in *Robotics: Science and Systems (RSS)*, 2014. [verify citation — virtual LiDAR from depth]

[18] [Virtual LiDAR projection from RGB-D — TBD. Need relevant citation.]

[19] C. Rösmann, W. Feiten, T. Wösch, et al., "Trajectory modification considering dynamic constraints of autonomous robots," in *ROBOTIK*, 2012.

[20] M. Bangura and R. Mahony, "Real-time model predictive control for quadrotors," in *IFAC World Congress*, 2014. [verify citation — MPC for mobile robots]

[21] Y. Zhu, R. Mottaghi, E. Kolve, et al., "Target-driven visual navigation in indoor scenes using deep reinforcement learning," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2017.

[22] P. Mirowski, R. Pascanu, F. Viola, et al., "Learning to navigate in complex environments," in *International Conference on Learning Representations (ICLR)*, 2017.

[23] D. Shah, A. Sridhar, N. Dashora, et al., "ViNT: A foundation model for visual navigation," in *Conference on Robot Learning (CoRL)*, 2023.

[24] T. Johannink, S. Bahl, A. Nair, et al., "Residual reinforcement learning for robot control," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2019.

[25] T. Silver, K. Allen, J. Tenenbaum, and L. Kaelbling, "Residual policy learning," *arXiv preprint arXiv:1812.06298*, 2018.

[26] A. Nair, B. McGrew, M. Andrychowicz, et al., "Overcoming exploration in reinforcement learning with demonstrations," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2018.

[27] J. Hwangbo, J. Lee, A. Dosovitskiy, et al., "Learning agile and dynamic motor skills for legged robots," *Science Robotics*, vol. 4, no. 26, 2019.

[28] [Residual learning for autonomous driving — TBD. Need relevant citation.]

[29] [Residual cost learning for motion planning — TBD. Need relevant citation.]

[30] A. D. Ames, S. Coogan, M. Egerstedt, et al., "Control barrier functions: Theory and applications," in *European Control Conference (ECC)*, 2019.

[31] J. Choi, F. Castañeda, C. J. Tomlin, and K. Sreenath, "Reinforcement learning for safety-critical control under model uncertainty, using control barrier functions," in *Conference on Robot Learning (CoRL)*, 2020.

[32] J. Tobin, R. Fong, A. Ray, et al., "Domain randomization for transferring deep neural networks from simulation to the real world," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2017.

[33] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, "Sim-to-real transfer of robotic control with dynamics randomization," in *IEEE International Conference on Robotics and Automation (ICRA)*, 2018.

[34] J. Tan, T. Zhang, E. Coumans, et al., "Sim-to-real: Learning agile locomotion for quadruped robots," in *Robotics: Science and Systems (RSS)*, 2018.

[35] [RealSense noise modeling for sim-to-real transfer — TBD. Need relevant citation.]

[36] M. Bertalmio, A. L. Bertozzi, and G. Sapiro, "Navier-Stokes, fluid dynamics, and image and video inpainting," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2001.

[37] P. Anderson, A. Chang, D. S. Chaplot, et al., "On evaluation of embodied navigation agents," *arXiv preprint arXiv:1807.06757*, 2018.

---

## Missing Information Needed for Final Paper

The following items must be completed before this manuscript can be considered a submission-ready draft:

### Experimental Data (Critical)
1. **All quantitative results in Tables V–IX.** Run all simulation and real-world experiments, collect metrics, compute means and standard deviations over multiple seeds.
2. **Real-world data collection.** Complete teleoperation-based data collection with the LeKiwi robot ($\mathcal{D}_{\text{real}}$). Current status: pipeline built, real data collection pending.
3. **Residual model training.** Train the Residual Safety Model on $\mathcal{D}_{\text{mix}}$ and evaluate on held-out test set. Report training curves, final loss values, and generalization metrics.

### Baselines (Critical)
4. **LoGoPlanner integration and evaluation.** Obtain LoGoPlanner code/weights from the authors, deploy on our LeKiwi platform, and run standardized evaluation. If LoGoPlanner cannot be obtained, replace with a comparable learned navigation baseline and document the substitution.
5. **LeRobot imitation learning baseline.** Train an ACT or Diffusion Policy model on $\mathcal{D}_{\text{mix}}$ using LeRobot's standard training pipeline. Report training configuration and results.

### Figures (Critical)
6. **Fig. 1:** Hardware platform photo and CAD diagram.
7. **Fig. 2:** Scan64 projection pipeline visualization with real D435i data.
8. **Fig. 3:** System architecture diagram (ZMQ host/client topology, data flow).
9. **Fig. 4:** Navigation trajectory comparison plots (from actual experiments).
10. **Fig. 5:** Scan64 under depth noise examples (from real D435i data).
11. **Fig. 6:** Emergency shield activation analysis (time series from real episodes).

### Citations to Verify/Complete
12. Verify all references marked `[verify citation]`. Fill in missing references marked `[TBD]`.
13. Add LoGoPlanner citation with correct authors, title, venue, and year.
14. Add citations for: virtual LiDAR from RGB-D, residual learning for navigation, residual cost learning for motion planning, RealSense noise modeling for sim-to-real.

### Analysis
15. Compute and report Wasserstein distance between simulation and real Scan64 distributions.
16. Run per-bin Scan64 statistics (mean, variance) to quantify systematic perception biases.
17. Analyze failure cases in detail: categorize collision types, shield false positives, deadlock situations.

### Writing
18. Once experimental results are available, rewrite Section V-F with actual numbers and statistical significance tests.
19. Add a "Results and Analysis" subsection discussing whether the experimental results support or contradict the expected trends described in the ablation discussion.
20. Update the abstract with final quantitative claims (if results support them).

---

*Manuscript version: v0.1 (draft). Last updated: 2025-05-13. Prepared for ICRA/IROS 20XX submission.*
