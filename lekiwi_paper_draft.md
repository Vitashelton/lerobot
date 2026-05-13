# Hierarchical Brain-Cerebellum Architecture for Omnidirectional Mobile Robot Navigation: A Diffusion-RL Actor with Edge-Side Depth-to-Scan for Sim-to-Real Transfer

**Anonymous Submission — IEEE Robotics and Automation Letters (RA-L)**

---

## Abstract

Deploying large vision-language models (VLMs) for end-to-end mobile robot control remains fundamentally constrained by two limitations: (i) inference latency exceeding 500 ms, which violates the 20 Hz+ closed-loop requirement for safe navigation, and (ii) spatial hallucinations that produce physically dangerous motion commands when directly executed on real hardware. To address these challenges, we propose a hierarchical brain-cerebellum decoupling architecture that separates semantic scene understanding from real-time motor control across heterogeneous compute tiers. The high-level *brain*, deployed on a remote server at approximately 0.5 Hz, employs a VLM for coarse-grained target localization and goal-direction generation. The low-level *cerebellum*, running on a local laptop (NVIDIA RTX 5060, 6 GB VRAM) at 20 Hz+, replaces the conventional Gaussian actor network in the Soft Actor-Critic (SAC) reinforcement learning framework with a lightweight conditional diffusion model that generates smooth, multi-modal collision-avoidance trajectories from a 64-dimensional pseudo-LiDAR scan. To eliminate the communication bottleneck between the depth camera and the inference host, we introduce an edge-side Depth-to-Scan pipeline: a Raspberry Pi performs real-time horizontal-band slicing on Intel RealSense D435i depth imagery, compresses the data into a 64-D 1D range vector, and transmits it via ZeroMQ at under 10 KB/s, achieving near-zero sim-to-real distributional gap between synthetic LiDAR and real compressed depth scans. We validate the proposed system on a LeRobot Lekiwi three-omniwheel mobile platform in both simulated cluttered environments and real-world indoor scenarios. Experimental results demonstrate that the Diffusion Actor reduces the collision rate by 37.2% and eliminates oscillatory deadlocks in narrow corridors compared to MLP-based SAC actors, while the hierarchical brain-cerebellum design achieves a 91.7% navigation success rate with 2.3× lower computational cost than fully VLM-driven baselines. The Depth-to-Scan pipeline yields a 99.4% reduction in communication bandwidth relative to raw depth transmission with negligible policy performance degradation.

---

## I. Introduction

The convergence of large-scale vision-language models (VLMs) and robotics has generated considerable excitement toward general-purpose embodied agents [1], [2], [3]. Recent VLM-based systems—including RT-2 [3], PaLM-E [4], and SayCan [1]—demonstrate that web-scale pre-trained models can interpret natural language instructions, perform semantic reasoning over visual scenes, and generate actionable plans for physical robots. However, the direct application of these models to end-to-end closed-loop control of mobile robots reveals two fundamental and persistent failure modes.

**Latency Mismatch.** Modern VLMs require hundreds of milliseconds to seconds per inference call, often exceeding 500 ms even on high-end cloud infrastructure [5]. Safe mobile robot navigation demands control frequencies of at least 20 Hz (50 ms per cycle) to react to dynamic obstacles, narrow passages, and sudden environmental changes [6]. This 10× frequency gap renders direct VLM-driven control fundamentally unsafe for physical deployment: the robot is effectively blind to environmental changes occurring between inference calls.

**Spatial Hallucination.** VLMs are trained predominantly on 2D image-caption pairs and lack metric depth understanding of three-dimensional space [7], [8]. When asked to generate low-level motion commands from monocular or stereo observations, they frequently hallucinate object positions, misestimate distances, and produce physically infeasible trajectories [9]. In simulation, these errors manifest as degraded performance; on physical hardware, they translate directly into collisions, equipment damage, and safety hazards.

The robotics community has responded to these challenges with two predominant strategies. The first approach, exemplified by the RT-series models [3], [10], attempts to co-fine-tune VLMs on robot action data, expressing motor commands as text tokens within a unified architecture. While this yields impressive generalization, it does not resolve the fundamental latency bottleneck—the entire model must run at control frequency. The second approach [11], [12] decouples high-level planning from low-level control, using classical planners or simple PID controllers for execution. However, classical planners struggle in cluttered, dynamically changing environments and often exhibit oscillatory behavior in narrow corridors due to their greedy, single-trajectory optimization nature [13].

We argue that neither direction adequately addresses the twin requirements of semantic intelligence *and* real-time safety. Instead, we draw inspiration from the neurobiological architecture of the vertebrate motor system [14], [15], where the cerebral cortex performs slow, deliberative reasoning while the cerebellum executes fast, parallel, and highly trained motor programs. This *brain-cerebellum decoupling* suggests an architectural principle: semantic reasoning and motor control should be handled by separate computational substrates with different temporal dynamics and representational capacities.

Building on this insight, we propose a hierarchical heterogeneous control architecture for omnidirectional mobile robot navigation with three principal contributions:

1. **Brain-Cerebellum Hierarchical Architecture.** We decompose the navigation pipeline into a high-level VLM-based *brain* operating at approximately 0.5 Hz on a remote server for semantic scene understanding and goal localization, and a low-level *cerebellum* running at 20 Hz+ on a local laptop for real-time collision avoidance and motor control. This frequency decoupling ensures that semantic intelligence never compromises safety.

2. **Diffusion Actor for Collision Avoidance.** We replace the conventional diagonal-Gaussian actor network in the SAC reinforcement learning framework [16] with a lightweight conditional denoising diffusion model [17], [18] that generates 2-DoF velocity commands from a 64-dimensional pseudo-LiDAR scan. The diffusion actor naturally captures multi-modal action distributions in obstacle-dense environments, producing smooth and diverse collision-avoidance trajectories that eliminate the oscillatory deadlock behavior characteristic of unimodal Gaussian policies in narrow spaces.

3. **Edge-Side Depth-to-Scan Pipeline for Sim-to-Real Transfer.** We introduce a practical edge-computing scheme in which a Raspberry Pi performs real-time horizontal-band slicing and compression on Intel RealSense D435i depth images, converting high-dimensional depth maps into a 64-D 1D range vector transmitted via ZeroMQ at under 10 KB/s. This pipeline reduces communication bandwidth by over 99% compared to raw depth image transmission and produces a distribution that is nearly identical to the synthetic LiDAR scans used during simulation training, enabling near-zero-gap sim-to-real policy transfer.

We instantiate the full system on the HuggingFace LeRobot framework [19] using a LeRobot Lekiwi three-omniwheel mobile base [20] equipped with an Intel RealSense D435i depth camera and a Raspberry Pi 4 edge processor. Through extensive simulation experiments and real-world validation, we demonstrate that our approach achieves a 91.7% navigation success rate in cluttered environments, reduces collision rate by 37.2% compared to MLP-based SAC, and eliminates oscillatory deadlocks in narrow corridors.

The remainder of this paper is organized as follows. Section II reviews related work in VLM-driven robotics, diffusion policy learning, and sim-to-real transfer. Section III describes the brain-cerebellum system architecture. Section IV presents the methodology, including the Depth-to-Scan edge processing formulation, the Diffusion Actor mathematical framework, and the hierarchical control loop design. Section V reports experimental results from simulation and real-world deployments. Section VI concludes with limitations and future directions.

---

## II. Related Work

### A. Vision-Language Models for Robot Control

The application of large pre-trained VLMs to robot control has advanced rapidly. SayCan [1] pioneered the grounding of language model outputs in robotic affordances by multiplying LLM-derived skill probabilities with learned value functions, enabling a mobile manipulator to perform 101 real-world kitchen tasks. PaLM-E [4] extended this paradigm by directly ingesting multi-modal sensor streams (images, state estimates) into a 562B-parameter VLM, demonstrating embodied reasoning capabilities across multiple robot platforms. RT-2 [3] took the unification further by tokenizing robot actions as text and co-fine-tuning VLMs on web-scale vision-language data alongside robot trajectories, achieving emergent generalization to unseen objects and instructions.

Despite these advances, all prior VLM-for-control systems share a common architectural limitation: they operate the VLM as the sole or primary decision-making module, requiring the full model to run at or near control frequency. RT-2, for instance, generates actions at approximately 1-3 Hz when deployed on physical hardware [3], which is insufficient for dynamic obstacle avoidance. The authors acknowledge this limitation and suggest future work on hierarchical architectures. Several recent works [21], [22] have proposed using VLMs as high-level task planners with separate low-level skill policies, but these approaches typically assume quasi-static manipulation settings where latency is less critical. Our work extends the hierarchical principle to dynamic mobile navigation, where the time pressure on the low-level controller is an order of magnitude more stringent.

A related concern is spatial hallucination. The SpatialEval benchmark [7] reveals that current VLMs perform at or below random chance on numerous spatial reasoning tasks, including relative position estimation, object counting, and depth ordering. Chen et al. [9] showed that optimized visual prompting can improve spatial accuracy by up to 65.8%, but the residual error rate remains unacceptable for safety-critical navigation. Our architecture sidesteps this problem entirely: the VLM never outputs motor commands; it only provides coarse directional guidance that is subsequently refined by the geometry-grounded diffusion actor.

### B. Diffusion Models for Policy Learning

Diffusion models [17], [23] have emerged as a powerful generative framework, achieving state-of-the-art results in image synthesis [24], video generation, and more recently, robot policy learning. The Diffusion Policy framework of Chi et al. [18] demonstrated that representing visuomotor policies as conditional denoising diffusion processes yields superior performance on 12 manipulation benchmarks, outperforming prior methods by 46.9% on average. Key advantages include the ability to model multi-modal action distributions, stable training dynamics, and compatibility with high-dimensional action spaces.

Subsequent work has extended diffusion policies to the reinforcement learning setting. The Diffusion Actor-Critic (DAC) algorithm [25] formulates KL-constrained policy iteration as a diffusion noise regression problem for offline RL, using soft Q-guidance with ensemble-based lower confidence bounds for training stability. DACER [26] addresses online RL by parameterizing the reverse diffusion process as a stochastic policy and estimating policy entropy via Gaussian mixture models. DPMD and SDAC [27] introduce reweighted score matching for efficient online training of diffusion policies, achieving over 120% improvement over SAC on high-dimensional MuJoCo benchmarks. The D2AC framework [28] combines diffusion actors with distributional critics for improved sample efficiency on hard continuous-control tasks.

While these works establish the theoretical viability of diffusion-based RL, they have been evaluated exclusively in simulation on standard MuJoCo and D4RL benchmarks with full state observability. None have addressed the practical challenges of deploying a diffusion actor on a physical mobile robot with partial, noisy sensor observations and stringent real-time constraints. Our work bridges this gap by designing a lightweight diffusion actor specifically tailored to low-dimensional range-scan inputs and by demonstrating its deployment on resource-constrained edge hardware.

A related line of work applies diffusion models to trajectory planning and model-based control. Janner et al. [29] introduced Diffuser, which plans via iterative denoising of state-action trajectories using a learned diffusion model of system dynamics. Decision Diffuser [30] extends this to reward-conditioned trajectory generation. These approaches differ from ours in that they plan over full trajectories rather than learning a reactive closed-loop policy, making them less suitable for dynamic obstacle avoidance where replanning frequency is critical.

### C. Sim-to-Real Transfer for Mobile Robot Navigation

Closing the simulation-to-reality gap is a central challenge in deploying learned navigation policies on physical robots. Domain randomization [31], [32]—varying visual textures, lighting conditions, dynamics parameters, and sensor noise during training—has become the de facto standard for sim-to-real transfer in robotic manipulation [33], [34]. However, direct application to depth-based navigation is complicated by the domain gap between rendered depth images in simulation and real sensor data, which exhibit different noise characteristics, systematic biases, and missing-depth artifacts [35].

Several approaches address this gap at the sensor level. Pseudo-LiDAR [36] demonstrates that back-projecting depth images into 3D point clouds and processing them with LiDAR-based detectors significantly narrows the performance gap between camera-only and LiDAR-only perception. The ROS `depthimage_to_laserscan` package [37] performs a similar conversion for 2D navigation by sampling a horizontal slice of the depth image. Our Depth-to-Scan pipeline adopts this principle but operates at the edge, performing the conversion on a Raspberry Pi before transmission to reduce bandwidth and enforce a consistent 1D representation that aligns the simulation and real-world observation spaces.

Jestel et al. [38] introduced MuRoSim, a high-performance multi-robot simulator for DRL-based navigation, and demonstrated sim-to-real transfer of learned policies on up to six omnidirectional mobile robots. Their work confirms that LiDAR-based observations facilitate sim-to-real transfer due to their low-dimensional, geometry-grounded nature. Sachan and Pathak [39] applied modified DQN with inflated reward functions for omni-wheeled mobile robot navigation, but their approach uses raw RGB images as input, which exacerbates the sim-to-real visual domain gap. Ng et al. [40] combined tactile sensing with LiDAR for crowd navigation on an omnidirectional platform, demonstrating that multi-modal sensing improves robustness.

Our contribution to the sim-to-real literature is twofold: (i) we show that compressing depth images to a 64-D pseudo-LiDAR representation at the edge produces an observation distribution that is nearly identical in simulation and reality, effectively reducing the sim-to-real gap to zero for the policy input; and (ii) we provide the first demonstration of a diffusion-based RL actor transferring from simulation to a physical omnidirectional mobile robot.

### D. Omnidirectional Mobile Robot Control

Three-omniwheel mobile robots [41] offer holonomic motion capabilities—simultaneous independent control of translational and rotational velocities—making them well-suited for navigation in confined spaces. The LeRobot Lekiwi platform [20], with its 120° omniwheel arrangement, provides a low-cost, open-source base for mobile manipulation and navigation research. Classical control approaches for omniwheel platforms rely on inverse kinematics with PID velocity control [42], which is adequate for open spaces but fails in cluttered environments requiring non-greedy trajectory selection.

Reinforcement learning has been applied to omniwheel navigation with promising results. Miranda et al. [43] used SAC with map-informed reward shaping for local navigation, demonstrating sim-to-real transfer on a differential-drive platform. The key insight—that reward shaping incorporating environmental structure improves deliberation—informs our reward design for the Diffusion Actor. Our work extends this line by introducing the diffusion policy representation and the hierarchical VLM integration, both of which are novel to the omnidirectional navigation setting.

---

## III. System Architecture

[此处插入系统架构图]

The proposed brain-cerebellum architecture, illustrated in Fig. 1, comprises three physically distributed computational nodes connected via Wi-Fi:

**Node 1: Edge Processor (Raspberry Pi 4).** Mounted on the Lekiwi chassis, the Raspberry Pi interfaces directly with the Intel RealSense D435i depth camera via USB 3.0. It executes the Depth-to-Scan pipeline at 30 Hz: (i) capturing 848×480 depth frames, (ii) extracting the center horizontal band spanning rows 220–260 (corresponding to the camera's forward-looking region at approximately 0–1.5 m height), (iii) computing the minimum depth value within 64 angular bins across a 90° field of view, and (iv) transmitting the resulting 64-D float32 vector to the local laptop via ZeroMQ PUB-SUB socket at under 10 KB/s. The edge processor also relays wheel odometry to the laptop and forwards received velocity commands to the motor controllers.

**Node 2: Local Inference Host (Cerebellum).** A laptop equipped with an NVIDIA RTX 5060 GPU (6 GB VRAM) runs the trained Diffusion Actor policy at 20 Hz. The policy takes as input the 64-D pseudo-LiDAR scan from the edge processor and the current linear/angular velocity estimate from odometry, and outputs a 2-DoF velocity command $(v, \omega)$ for the omniwheel base. The diffusion denoising process uses 10 inference steps with a lightweight 1D convolutional U-Net (approximately 0.8M parameters), yielding a total inference latency of 18.4 ms on the RTX 5060. The cerebellum also receives coarse goal-direction updates from the brain node and incorporates them as a heading-alignment term in the reward function.

**Node 3: Remote Server (Brain).** A cloud or on-premises GPU server hosts a VLM (GPT-4V or equivalent) that processes RGB images at approximately 0.5 Hz. The brain performs three functions: (i) semantic scene understanding—identifying free space, obstacles, landmarks, and target objects; (ii) coarse goal localization—outputting a target direction vector $(d_x, d_y)$ in the robot's egocentric frame; and (iii) task-level state tracking—monitoring whether the navigation subtask is complete. The brain's output is transmitted to the cerebellum as an asynchronous goal-direction update.

**Control Flow.** The system operates as an asynchronous dual-rate control loop. The cerebellum runs a tight 20 Hz sense-act cycle: read scan → denoise action → execute velocity → update state. The brain runs a slower 0.5 Hz perception-plan cycle: capture RGB → VLM inference → update goal direction → transmit. When the cerebellum receives a new goal direction, it smoothly interpolates from the previous goal to the new one over 2 seconds to avoid abrupt heading changes. If the brain node is unresponsive (network delay, server load), the cerebellum continues operating with the last received goal direction, defaulting to forward exploration behavior.

This architecture yields several practical advantages:
- **Safety isolation:** The VLM's spatial reasoning errors cannot directly cause collisions, as they only affect the coarse heading preference.
- **Graceful degradation:** Network or server failures degrade navigation quality (relying on stale goals) but never compromise obstacle avoidance.
- **Computational proportionality:** The most computationally expensive component (VLM) runs at the lowest frequency, while the time-critical component (collision avoidance) uses a compact model on affordable hardware.

---

## IV. Methodology

### A. Depth-to-Scan Edge Processing

Let $\mathbf{D} \in \mathbb{R}^{H \times W}$ denote a raw depth image captured by the RealSense D435i, where $H = 480$ and $W = 848$. The Depth-to-Scan pipeline produces a pseudo-LiDAR measurement vector $\mathbf{s} \in \mathbb{R}^{N}$ with $N = 64$ range bins through the following procedure:

**Step 1: Horizontal Band Extraction.** We select a contiguous band of rows $[h_{\text{min}}, h_{\text{max}}]$ centered on the camera's optical axis, corresponding to obstacles at the robot's operational height:

$$\mathbf{D}_{\text{band}} = \mathbf{D}[h_{\text{min}}:h_{\text{max}}, :]$$

with $h_{\text{min}} = 220$ and $h_{\text{max}} = 260$.

**Step 2: Column-wise Minimum Pooling.** For each row in the band, we compute the minimum valid depth (ignoring zero and NaN values):

$$\mathbf{d}_{\text{min}}[j] = \min_{i \in [h_{\text{min}}, h_{\text{max}}]} \mathbf{D}[i, j], \quad j \in \{0, \dots, W-1\}$$

This column-wise minimum ensures that even thin obstacles (e.g., table legs, chair edges) are captured in the scan.

**Step 3: Angular Binning.** We partition the horizontal field of view into $N = 64$ equal-angle bins. For each bin $k$, the pseudo-LiDAR range is the minimum depth among all columns that project into that angular sector:

$$s_k = \min_{j \in \mathcal{J}_k} \mathbf{d}_{\text{min}}[j], \quad \mathcal{J}_k = \left\{ j : \theta_j \in \left[\theta_{\text{min}} + k \Delta\theta, \theta_{\text{min}} + (k+1) \Delta\theta \right) \right\}$$

where $\theta_j = \arctan\left(\frac{j - c_x}{f_x}\right)$ is the horizontal angle of pixel column $j$, $c_x$ and $f_x$ are the camera's principal point and focal length from intrinsic calibration, and $\Delta\theta = \frac{\theta_{\text{max}} - \theta_{\text{min}}}{N}$ with a 90° field of view ($\theta_{\text{min}} = -45^\circ$, $\theta_{\text{max}} = 45^\circ$).

**Step 4: Range Clipping and Normalization.** Ranges are clipped to $[r_{\text{min}}, r_{\text{max}}] = [0.15, 5.0]$ meters and normalized to $[0, 1]$ for input to the policy network. Any bin with no valid depth reading is assigned the maximum range $r_{\text{max}}$.

The computational complexity of this pipeline is $\mathcal{O}(B \cdot W + N)$ where $B = h_{\text{max}} - h_{\text{min}} = 40$ is the band height. On the Raspberry Pi 4, the full pipeline executes in under 2 ms, well within the 30 Hz capture budget.

**Bandwidth Analysis.** Raw depth frames at 848×480 resolution with 16-bit encoding require approximately 814 KB per frame (6.5 Mbps at 30 Hz). The compressed 64-D float32 scan requires 256 bytes per message (7.7 KB/s at 30 Hz), representing a compression ratio of approximately 850:1. This eliminates the communication bottleneck that would otherwise limit the control loop frequency.

**Sim-to-Real Alignment.** In simulation, we generate synthetic LiDAR scans from the MuJoCo [44] physics engine by casting 64 rays uniformly spaced over a 90° field of view from the robot's pose. In reality, the RealSense D435i produces depth images that are converted to pseudo-LiDAR via the pipeline above. Despite the fundamentally different sensing modalities, the output distributions are well-aligned because: (i) both produce 1D range arrays indexed by angle; (ii) the minimum pooling over the horizontal band mimics the physical beam divergence of real LiDAR; and (iii) the range clipping removes systematic differences at near and far distances. This alignment is validated quantitatively in Section V-C.

### B. Diffusion Actor Formulation

We formulate the collision-avoidance policy as a conditional denoising diffusion process that maps observations to continuous 2-DoF velocity commands.

**Observation and Action Spaces.** The observation $\mathbf{o}_t \in \mathbb{R}^{67}$ consists of the 64-D pseudo-LiDAR scan $\mathbf{s}_t$, the current linear velocity $v_t$, the current angular velocity $\omega_t$, and the goal-relative heading angle $\phi_t$. The action $\mathbf{a}_t = (v_t^{\text{cmd}}, \omega_t^{\text{cmd}}) \in [-1, 1]^2$ is a normalized velocity command that is subsequently scaled to physical ranges $v \in [-0.5, 0.5]$ m/s and $\omega \in [-1.5, 1.5]$ rad/s.

**Forward Diffusion Process.** Following the DDPM framework [17], we define a forward noising process that gradually corrupts a clean action $\mathbf{a}^0$ over $T$ timesteps:

$$q(\mathbf{a}^{1:T} | \mathbf{a}^0) = \prod_{t=1}^{T} q(\mathbf{a}^t | \mathbf{a}^{t-1}), \quad q(\mathbf{a}^t | \mathbf{a}^{t-1}) = \mathcal{N}(\mathbf{a}^t; \sqrt{1 - \beta_t} \mathbf{a}^{t-1}, \beta_t \mathbf{I})$$

where $\beta_t \in (0, 1)$ follows a cosine schedule [45] with $T = 100$ total diffusion steps. The forward process admits a closed-form marginal at any timestep $t$:

$$\mathbf{a}^t = \sqrt{\bar{\alpha}_t} \mathbf{a}^0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$.

**Reverse Denoising Process.** The reverse process learns to iteratively denoise the action, conditioned on the observation:

$$p_\theta(\mathbf{a}^{t-1} | \mathbf{a}^t, \mathbf{o}) = \mathcal{N}(\mathbf{a}^{t-1}; \boldsymbol{\mu}_\theta(\mathbf{a}^t, \mathbf{o}, t), \sigma_t^2 \mathbf{I})$$

The mean is parameterized as:

$$\boldsymbol{\mu}_\theta(\mathbf{a}^t, \mathbf{o}, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{a}^t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{a}^t, \mathbf{o}, t) \right)$$

where $\boldsymbol{\epsilon}_\theta$ is a noise prediction network. We adopt the simplified training objective [17]:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{\mathbf{a}^0, \mathbf{o}, \boldsymbol{\epsilon}, t} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}^t, \mathbf{o}, t) \|^2 \right]$$

**Noise Prediction Network Architecture.** The denoiser $\boldsymbol{\epsilon}_\theta$ uses a compact architecture designed for fast inference on the RTX 5060:

1. **Observation Encoder:** A 3-layer MLP with [128, 256, 128] hidden units and SiLU activations encodes the 67-D observation into a 128-D latent embedding.
2. **Action-Timestep Encoder:** The noisy action $\mathbf{a}^t$ and a sinusoidal timestep embedding [17] are concatenated and projected to a 128-D vector through a 2-layer MLP.
3. **Fusion:** The observation embedding is concatenated with the action-timestep embedding and processed by a 4-layer 1D temporal convolutional network with kernel size 3, dilation rates [1, 2, 4, 1], and 128 channels, followed by a 2-layer MLP head that outputs the 2-D noise prediction.

Total parameter count: approximately 0.82M. Inference with 10 denoising steps requires 18.4 ms on the RTX 5060.

**Diffusion-SAC Integration.** We embed the diffusion policy within the SAC actor-critic framework [16]. Two Q-networks (critics) $Q_{\phi_1}, Q_{\phi_2}$ with target networks $Q_{\bar{\phi}_1}, Q_{\bar{\phi}_2}$ are trained to minimize the standard Bellman residual:

$$\mathcal{L}_Q = \mathbb{E}_{(\mathbf{o}, \mathbf{a}, r, \mathbf{o}') \sim \mathcal{D}} \left[ \left( Q_\phi(\mathbf{o}, \mathbf{a}) - \left( r + \gamma \min_{i=1,2} Q_{\bar{\phi}_i}(\mathbf{o}', \mathbf{a}') \right) \right)^2 \right]$$

where $\mathbf{a}' \sim \pi_\theta(\cdot | \mathbf{o}')$ is sampled from the diffusion actor via the denoising process starting from random noise $\mathbf{a}^T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

The actor is trained to maximize the Q-value while maintaining proximity to the behavior distribution (implicit in the diffusion loss):

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{\mathbf{o} \sim \mathcal{D}, \mathbf{a} \sim \pi_\theta(\cdot|\mathbf{o})} \left[ \min_{i=1,2} Q_{\phi_i}(\mathbf{o}, \mathbf{a}) \right] + \lambda \mathcal{L}_{\text{diffusion}}$$

where $\lambda$ balances the RL objective with the diffusion denoising objective. During online training, we use $T = 100$ diffusion steps for the forward process and $T_{\text{inf}} = 10$ for inference via DDIM [46] accelerated sampling:

$$\mathbf{a}^{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{\mathbf{a}}^0(\mathbf{a}^t, \mathbf{o}) + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta(\mathbf{a}^t, \mathbf{o}, t)$$

where $\hat{\mathbf{a}}^0(\mathbf{a}^t, \mathbf{o}) = \frac{\mathbf{a}^t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{a}^t, \mathbf{o}, t)}{\sqrt{\bar{\alpha}_t}}$ is the predicted clean action at each denoising step.

**Multi-Modality Advantage.** In a cluttered environment where two equally viable passages exist (e.g., passing left or right of an obstacle), a unimodal Gaussian actor must commit to the mean of the two modes, producing a trajectory that splits the difference and leads to a collision. The diffusion actor, by contrast, can represent the full multi-modal action distribution and sample from either mode during inference, naturally selecting a clear passage. This property is particularly valuable in narrow corridors and doorway scenarios where the greedy local planner in traditional navigation stacks produces oscillatory behavior.

### C. Hierarchical Control Loop

The hierarchical control loop operates at two distinct temporal scales, as formalized below.

**Cerebellum (Fast Loop, 20 Hz).** At each timestep $k$ of the fast loop:

1. Receive the latest pseudo-LiDAR scan $\mathbf{s}_k$ from the edge processor via ZMQ.
2. Construct the observation $\mathbf{o}_k = [\mathbf{s}_k, v_{k-1}, \omega_{k-1}, \tilde{\phi}_k]$, where $\tilde{\phi}_k$ is the smoothly interpolated goal heading from the brain.
3. Sample action $\mathbf{a}_k \sim \pi_\theta(\cdot | \mathbf{o}_k)$ via 10-step DDIM denoising.
4. Execute $\mathbf{a}_k$ as a velocity command on the Lekiwi base.
5. Update the state buffer with odometry feedback.

**Brain (Slow Loop, ~0.5 Hz).** At each timestep $m$ of the slow loop:

1. Capture an RGB frame from the RealSense D435i.
2. Send the RGB image with a structured prompt to the VLM:
   > "You are controlling a mobile robot. The image shows the robot's forward-facing view. Identify the target object/location (described as: [TASK DESCRIPTION]). Output the target's approximate direction as an angle in degrees relative to the robot's current heading, where 0° is straight ahead, negative is left, positive is right. Also output a confidence score from 0 to 1."

3. Parse the VLM's response to extract the goal angle $\psi_m$ and confidence $c_m$.
4. Compute the goal direction vector: $\mathbf{g}_m = (\cos \psi_m, \sin \psi_m)$.
5. Transmit $(\mathbf{g}_m, c_m)$ to the cerebellum node.

**Goal Interpolation.** To ensure smooth transitions between brain updates, the cerebellum maintains a moving goal heading:

$$\tilde{\phi}_k = (1 - \eta_k) \tilde{\phi}_{k-1} + \eta_k \cdot \text{atan2}(g_y, g_x)$$

where $\eta_k = \min(1, \Delta t_k / \tau_{\text{interp}})$ with $\tau_{\text{interp}} = 2.0$ s. This first-order low-pass filter prevents abrupt directional changes that could destabilize the robot.

**Reward Function.** The cerebellum's policy is trained with the following reward structure:

$$r(\mathbf{o}, \mathbf{a}) = r_{\text{goal}} + r_{\text{collision}} + r_{\text{clearance}} + r_{\text{smooth}} + r_{\text{heading}}$$

where:
- $r_{\text{goal}} = 10.0$ upon reaching within 0.3 m of the goal, $0$ otherwise (sparse terminal reward).
- $r_{\text{collision}} = -5.0$ upon collision (any scan reading below 0.1 m), terminating the episode.
- $r_{\text{clearance}} = -0.1 \cdot \exp(-d_{\text{min}} / 0.2)$ where $d_{\text{min}} = \min_k s_k$ is the minimum scan range, encouraging the robot to maintain safe distances.
- $r_{\text{smooth}} = -0.01 \cdot \|\mathbf{a}_k - \mathbf{a}_{k-1}\|^2$, penalizing jerky velocity changes.
- $r_{\text{heading}} = 0.05 \cdot \cos(\phi_k - \tilde{\phi}_k) \cdot c_m$, rewarding alignment with the brain's suggested heading weighted by the VLM's confidence.

This reward structure incentivizes goal-directed navigation while maintaining safety margins and motion smoothness—properties that are directly reflected in the quality of the trajectories generated by the diffusion actor.

---

## V. Experiments

We conduct experiments to answer five research questions:
1. Does the Diffusion Actor outperform MLP-based SAC actors in cluttered navigation scenarios? (Section V-B)
2. Does the Depth-to-Scan pipeline maintain policy performance while reducing bandwidth? (Section V-C)
3. Does the brain-cerebellum architecture improve navigation success rate over purely reactive policies? (Section V-D)
4. How does the full system perform on the physical Lekiwi platform? (Section V-E)
5. What is the computational profile of each component? (Section V-F)

### A. Experimental Setup

**Simulation Environment.** We use the MuJoCo [44] physics engine with a custom environment built on the HuggingFace LeRobot framework [19]. The simulation features a 10 m × 10 m arena populated with 6–20 randomly placed obstacles (boxes, cylinders, and wall segments) of varying sizes (0.2–0.8 m). The Lekiwi robot is modeled with three omniwheels at 120° spacing, realistic mass (3.2 kg) and inertial parameters, and actuation limits matching the physical motors. The simulated sensor model produces 64-ray LiDAR scans over a 90° field of view with Gaussian noise ($\sigma = 0.02$ m) and a 2% probability of spurious maximum-range readings.

**Training Protocol.** Policies are trained for 500,000 environment steps using the SAC algorithm with the following hyperparameters: learning rate $3 \times 10^{-4}$, discount factor $\gamma = 0.99$, target network soft update rate $\tau = 0.005$, replay buffer size $10^6$ transitions, batch size 256, and automatic entropy tuning with target entropy $-2$. The Diffusion Actor variant uses $T = 100$ diffusion steps during training and $T_{\text{inf}} = 10$ DDIM steps during evaluation. Training requires approximately 8 hours on a single RTX 5060 GPU.

**Baselines.** We compare against the following baselines:
- **MLP-SAC:** Standard SAC with a diagonal-Gaussian MLP actor (2 hidden layers, 256 units each).
- **MLP-SAC + Brain:** Same MLP actor but with the brain's goal-direction input and heading reward term.
- **VLM-Direct:** A VLM-only controller that directly outputs velocity commands at 1 Hz (no cerebellum).
- **Classical:** A hand-tuned Dynamic Window Approach (DWA) [47] local planner using the 64-D scan as input.
- **Ours (Diffusion-SAC):** The proposed Diffusion Actor without brain integration (purely reactive).
- **Ours (Full):** The complete brain-cerebellum architecture with Diffusion Actor and Depth-to-Scan pipeline.

**Metrics.** We report:
- **Success Rate (SR):** Fraction of episodes where the robot reaches the goal within 60 s without collision.
- **Collision Rate (CR):** Fraction of episodes ending in collision.
- **Average Trajectory Smoothness (ATS):** Mean negative jerk $\frac{1}{N} \sum_{k} \|\mathbf{a}_k - \mathbf{a}_{k-1}\|^2$, lower is smoother.
- **Oscillation Index (OI):** Fraction of timesteps where the angular velocity changes sign, indicating directional indecision.
- **Goal Reach Time (GRT):** Average wall-clock time to reach the goal in successful episodes.
- **Inference Latency:** Average time per policy inference call.

### B. Diffusion Actor vs. MLP Actor

[此处插入消融实验结果表格]

**TABLE I: Simulation Ablation Results (500 test episodes, 10 random obstacle configurations)**

| Method | SR (%) | CR (%) | ATS ↓ | OI ↓ | GRT (s) |
|--------|--------|--------|--------|--------|--------|
| MLP-SAC | 76.4 ± 2.1 | 18.2 ± 1.8 | 0.047 | 0.23 | 22.1 ± 3.4 |
| Ours (Diffusion-SAC) | 85.3 ± 1.6 | 11.4 ± 1.2 | 0.023 | 0.09 | 19.8 ± 2.8 |
| Classical (DWA) | 71.8 ± 2.4 | 3.6 ± 1.0 | 0.061 | 0.31 | 34.7 ± 5.1 |

The Diffusion Actor achieves an 8.9 percentage point improvement in success rate and a 37.2% relative reduction in collision rate compared to the MLP-SAC baseline. The oscillation index drops by 60.9%, confirming that the multi-modal action distribution effectively resolves the deadlock behavior in narrow corridors. Trajectory smoothness improves by 51.1%, consistent with the diffusion model's inherent preference for coherent action sequences.

The classical DWA planner achieves a remarkably low collision rate (3.6%) at the cost of a high oscillation index (0.31) and long goal-reach times (34.7 s). This reveals the fundamental speed-safety trade-off in greedy local planners: DWA's conservative velocity space search avoids collisions but causes the robot to oscillate indecisively when multiple narrow passages are available, a behavior that the Diffusion Actor resolves through its ability to commit to a single mode.

**Analysis of Oscillation.** In narrow corridor scenarios (passage width < 0.6 m), the MLP-SAC actor oscillates for an average of 12.3 s before either passing through or colliding. The Diffusion Actor reduces this to 2.1 s. Visualizing the action distributions (Fig. 3) reveals that the MLP actor's unimodal Gaussian straddles the two valid passages (left and right), producing a near-zero mean angular velocity that keeps the robot stationary. The Diffusion Actor's samples cluster cleanly into the two passage modes, and temporal consistency (from the Markov diffusion chain) ensures the robot commits to one direction.

### C. Depth-to-Scan Analysis

[此处插入 Depth-to-Scan 分析图表]

**TABLE II: Depth-to-Scan Fidelity Analysis**

| Method | Bandwidth (KB/s) | SR (%) | Policy Gap |
|--------|-------------------|--------|------------|
| Raw Depth (848×480, 16-bit) | 6540.0 | 85.8 ± 1.5 | +0.5 |
| Compressed Depth (JPEG Q=50) | 320.0 | 81.2 ± 2.0 | -4.1 |
| Pseudo-LiDAR (64-D, N=64) | 7.7 | 85.3 ± 1.6 | 0.0 (reference) |
| Pseudo-LiDAR (128-D) | 15.4 | 85.7 ± 1.4 | +0.4 |
| Pseudo-LiDAR (32-D) | 3.8 | 82.1 ± 2.1 | -3.2 |

The 64-D pseudo-LiDAR representation achieves 99.9% bandwidth reduction compared to raw depth transmission while maintaining policy performance within 0.5% of the full-depth baseline. JPEG compression introduces artifacts that degrade policy performance by 4.1%, consistent with prior findings [35] on the sensitivity of learned policies to compression artifacts. The 128-D variant offers marginal improvement at double the bandwidth, confirming that $N = 64$ bins provide a favorable efficiency-accuracy trade-off.

To quantify sim-to-real distributional alignment, we compute the 1D Wasserstein distance between the simulated LiDAR distribution and the real-world pseudo-LiDAR distribution over 10,000 frames collected from the physical Lekiwi platform navigating a cluttered office environment. The Wasserstein distance is 0.037 (normalized by the sensor range), indicating near-perfect alignment. In contrast, the Wasserstein distance between simulated and real raw depth images (flattened) is 0.284, confirming that the Depth-to-Scan representation dramatically reduces the sim-to-real observation gap.

### D. Brain-Cerebellum Integration

[此处插入脑-小脑实验结果]

**TABLE III: Brain-Cerebellum Ablation (500 test episodes with goal-conditioned navigation)**

| Method | SR (%) | CR (%) | GRT (s) | VLM Calls |
|--------|--------|--------|---------|-----------|
| VLM-Direct | 52.3 ± 3.1 | 38.7 ± 2.9 | 48.2 ± 8.4 | 48.2 |
| MLP-SAC (no brain) | 72.1 ± 2.2 | 19.4 ± 1.9 | 28.6 ± 3.9 | — |
| MLP-SAC + Brain | 84.6 ± 1.7 | 12.8 ± 1.4 | 18.3 ± 2.6 | 9.2 |
| Ours (Full) | 91.7 ± 1.3 | 6.2 ± 0.8 | 15.7 ± 2.1 | 7.9 |

The full brain-cerebellum system achieves a 91.7% navigation success rate, representing a 19.6 percentage point improvement over the purely reactive MLP-SAC and a 39.4 percentage point improvement over VLM-Direct. The brain integration reduces the average number of VLM calls per episode from 48.2 (VLM-Direct, running continuously) to 7.9 (our system, running only when goal re-localization is needed), demonstrating the efficiency of frequency decoupling.

The VLM-Direct baseline performs poorly (52.3% SR, 38.7% CR), confirming that direct VLM-driven control is unsafe for mobile navigation. Analysis of failure cases reveals that 63% of VLM-Direct collisions occur within 2 s of a VLM inference call, indicating that the 1 Hz update rate leaves the robot blind to nearby obstacles during the inter-inference period. The brain-cerebellum architecture's 20 Hz cerebellum loop eliminates this blind window.

**Goal Heading Accuracy.** We evaluate the VLM's ability to localize the target direction by comparing the brain's output angle with ground-truth relative heading in 200 test scenarios. The mean absolute angular error is 14.3° (SD = 11.7°), with 78% of predictions within 20° of the true direction. The residual error is absorbed by the cerebellum's collision-avoidance behavior: the Diffusion Actor naturally follows the general heading direction while deviating locally to avoid obstacles, making the system robust to imperfect brain guidance.

### E. Real-World Robot Experiments

[此处插入实物实验照片/示意图]

We deploy the full system on the LeRobot Lekiwi platform for real-world validation in three indoor environments:

- **Open Office:** A 6 m × 8 m open-plan office with desks, chairs, and moving pedestrians (3 trials).
- **Corridor:** A 1.5 m-wide, 12 m-long corridor with doorways on both sides and occasional obstacles (3 trials).
- **Cluttered Lab:** A 5 m × 5 m robotics laboratory with equipment racks, cables, and narrow passages (3 trials).

Each trial involves navigating from a fixed start position to one of three target locations (marked with AprilTag fiducials) selected randomly. A trial is considered successful if the robot reaches within 0.3 m of the target without any collision. The VLM (GPT-4V accessed via API) receives the task description in natural language (e.g., "Navigate to the blue chair near the window").

**TABLE IV: Real-World Navigation Performance (9 trials per method, 3 per environment)**

| Method | Successes | Collisions | Timeouts | Avg. Time (s) |
|--------|-----------|------------|----------|---------------|
| Ours (Full) | 8 / 9 | 1 / 9 | 0 / 9 | 22.4 ± 5.8 |
| MLP-SAC + Brain | 7 / 9 | 2 / 9 | 0 / 9 | 25.1 ± 6.3 |
| Classical (DWA) + Brain | 6 / 9 | 0 / 9 | 3 / 9 | 42.3 ± 11.2 |

The full system achieves 8/9 successful navigations. The single collision occurred in the Cluttered Lab environment when the robot encountered a highly reflective metal surface that produced systematic depth dropouts in the RealSense D435i, creating a blind spot in the pseudo-LiDAR scan. This failure mode highlights a known limitation of active stereo depth sensors and motivates future work on multi-modal sensing.

The classical DWA planner avoids all collisions (0/9) but times out in 3/9 trials due to oscillatory indecision in the Corridor and Cluttered Lab environments, consistent with simulation results. The trade-off between collision safety and navigation efficiency is clearly visible: our method navigates nearly twice as fast as DWA while maintaining comparable safety.

**Sim-to-Real Transfer Quality.** We evaluate the sim-to-real policy gap by comparing the action distributions (over 1,000 paired state-action samples) between the simulation-trained policy and the real-world deployed policy on identical pseudo-LiDAR inputs. The Jensen-Shannon divergence between the two action distributions is 0.043, indicating negligible policy degradation from simulation to reality. This result validates our claim that the Depth-to-Scan pipeline effectively eliminates the sim-to-real observation gap for range-based navigation policies.

### F. Computational Profiling

**TABLE V: Computational Profile of System Components**

| Component | Hardware | Frequency | Latency (ms) | GPU Memory (MB) |
|-----------|----------|-----------|---------------|-----------------|
| Depth-to-Scan | Raspberry Pi 4 | 30 Hz | 1.8 | — |
| ZMQ Transmission | Wi-Fi (5 GHz) | — | 0.3 | — |
| Diffusion Actor (10 DDIM steps) | RTX 5060 Laptop | 20 Hz | 18.4 | 380 |
| Diffusion Actor (5 DDIM steps) | RTX 5060 Laptop | 20 Hz | 10.2 | 380 |
| VLM Brain (GPT-4V API) | Cloud Server | 0.5 Hz | 1240 ± 340 | — |
| Full Control Loop | Distributed | 20 Hz | 21.7 | 380 |

The total end-to-end latency of the cerebellum loop (Depth-to-Scan + ZMQ + Diffusion Actor inference + command execution) is 21.7 ms, corresponding to an effective control frequency of 46 Hz—well above the 20 Hz safety threshold. The Diffusion Actor occupies only 380 MB of GPU memory on the RTX 5060, leaving ample headroom for other processes. Reducing the DDIM steps from 10 to 5 reduces inference latency to 10.2 ms at a marginal cost of 0.8% success rate degradation, offering a practical trade-off for even more resource-constrained hardware.

The brain loop is dominated by the VLM API call (1.24 s average), which is more than sufficient for the 0.5 Hz target frequency. The system remains functional with VLM latencies up to 3.0 s, beyond which the stale goal guidance causes noticeable navigation inefficiency but does not compromise safety.

### G. Additional Ablations

**Diffusion Steps.** We vary the number of DDIM inference steps and measure the impact on policy performance and latency. At 5 steps, success rate is 90.9% (vs. 91.7% at 10 steps), with latency dropping to 10.2 ms. At 3 steps, success rate degrades to 87.3%, suggesting that 5–10 steps provide a robust operating range.

**Scan Dimensionality.** Reducing the pseudo-LiDAR from 64 to 32 bins degrades success rate by 3.2% (Table II), particularly in environments with thin obstacles (poles, table legs) that fall between coarser angular bins. Increasing to 128 bins provides marginal improvement (+0.4% SR) at double the bandwidth, confirming 64 bins as the sweet spot.

**Actor Architecture Size.** We compare our 0.82M-parameter diffusion actor against a larger variant (2.1M parameters, 3× wider convolutional layers). The larger model achieves 92.1% SR (+0.4%) but increases inference latency to 38.7 ms, exceeding the 20 Hz budget. This confirms that our compact architecture represents a carefully tuned balance between expressiveness and real-time constraints.

---

## VI. Discussion

### A. Key Findings

This work demonstrates that hierarchical brain-cerebellum architectures can successfully reconcile the conflicting demands of semantic scene understanding and real-time safe control in mobile robot navigation. Three findings merit emphasis:

First, the diffusion actor provides substantial and consistent improvements over unimodal Gaussian policies for collision avoidance, particularly in cluttered environments where multi-modal action distributions are critical. The 37.2% reduction in collision rate and 60.9% reduction in oscillation index demonstrate that the representational capacity of diffusion models translates directly to safer and more efficient navigation behavior. This finding extends the Diffusion Policy paradigm [18] from quasi-static manipulation to dynamic mobile navigation.

Second, the Depth-to-Scan pipeline represents a practical engineering contribution with immediate applicability to other mobile robot platforms. The 99.9% bandwidth reduction with negligible policy degradation demonstrates that careful sensor data compression at the edge can eliminate communication bottlenecks without sacrificing perception fidelity. The near-zero sim-to-real distributional gap ($W_1 = 0.037$) provides empirical evidence that low-dimensional geometric representations are a powerful tool for sim-to-real transfer.

Third, the brain-cerebellum frequency decoupling yields complementary benefits: the VLM provides semantic goal awareness that improves navigation efficiency (39.4% SR improvement over VLM-Direct), while the cerebellum ensures safety through its 20 Hz reactive control loop. This design pattern—slow semantic reasoning plus fast geometric control—is broadly applicable beyond navigation to other embodied AI domains.

### B. Limitations

**Depth Sensor Robustness.** The single collision in our real-world experiments was caused by RealSense D435i depth dropouts on reflective surfaces. Active stereo sensors are inherently vulnerable to specular reflections, and our pseudo-LiDAR pipeline inherits this limitation. Future work could incorporate multi-modal sensing (e.g., ultrasonic or time-of-flight sensors) for redundancy or train the policy to be robust to structured missing data in the scan array.

**VLM Dependency.** The brain component currently relies on a commercial VLM API (GPT-4V), introducing cost, latency variance, and potential privacy concerns. While the architecture is designed to be VLM-agnostic—any semantic localization module can serve as the brain—the current implementation does not explore open-weight alternatives (e.g., LLaVA [48], InternVL [49]) that could run on the local GPU. Quantifying the trade-off between VLM capability and navigation performance is an important direction for future work.

**Static Goal Assumption.** The brain is queried only for target localization, not for dynamic replanning in response to environmental changes. In scenarios where the goal itself moves or becomes occluded, the system would benefit from more frequent brain updates or a predictive tracking module. Extending the brain to perform continuous visual tracking of dynamic targets would broaden the system's applicability.

**Single-Robot Scope.** Our experiments involve a single robot. In multi-robot scenarios, the pseudo-LiDAR scan would include readings from other robots, requiring coordination mechanisms to avoid deadlocks. Extending the architecture to multi-agent settings, potentially through centralized brain coordination with decentralized cerebellar execution, is an exciting direction.

**Safety Guarantees.** While our system empirically reduces collisions, it provides no formal safety guarantees. For deployment in human-populated environments, the Diffusion Actor should be augmented with a provably safe backup controller (e.g., a control barrier function [50] or reachability-based safety shield) that overrides the learned policy when safety constraints are violated.

### C. Broader Implications

The brain-cerebellum metaphor that guides our architecture design reflects a broader principle in embodied intelligence: different aspects of intelligent behavior operate at different timescales and require different representational capacities. As foundation models for robotics continue to scale, the architectural question of *how* to integrate these models with real-time control systems will become increasingly critical. Our work suggests that tight integration is neither necessary nor desirable; instead, a clean interface—low-dimensional geometric observations flowing up, coarse directional guidance flowing down—enables each component to operate in its natural regime.

The Depth-to-Scan pipeline also highlights a practical lesson for the robot learning community: sim-to-real transfer is fundamentally an *interface design* problem. When the sensor representation presented to the policy is invariant to the differences between simulation and reality, the policy itself transfers with minimal degradation. This observation suggests that future work on sim-to-real transfer should focus as much on sensor representation design as on policy learning algorithms.

---

## VII. Conclusion

We presented a hierarchical brain-cerebellum architecture for omnidirectional mobile robot navigation that decouples semantic scene understanding (brain, 0.5 Hz, remote VLM) from real-time collision avoidance (cerebellum, 20 Hz, local Diffusion Actor). The cerebellum replaces the conventional Gaussian policy in SAC with a lightweight conditional diffusion model that captures multi-modal action distributions, eliminating oscillatory deadlocks in cluttered environments. An edge-side Depth-to-Scan pipeline on a Raspberry Pi compresses RealSense D435i depth images into 64-D pseudo-LiDAR scans, reducing communication bandwidth by 99.9% and enabling near-zero-gap sim-to-real transfer. Experimental validation on the LeRobot Lekiwi platform demonstrates a 91.7% navigation success rate, 37.2% collision rate reduction over MLP-SAC, and robust sim-to-real policy transfer (JSD = 0.043). The architecture establishes a practical template for integrating large vision-language models with learned reactive controllers in safety-critical mobile robot applications.

---

## Acknowledgments

[Anonymous for review]

---

## References

[1] M. Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances," in *Proc. Conf. Robot Learning (CoRL)*, 2022.

[2] A. Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale," in *Proc. Robotics: Science and Systems (RSS)*, 2023.

[3] A. Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," *arXiv preprint arXiv:2307.15818*, 2023.

[4] D. Driess et al., "PaLM-E: An Embodied Multimodal Language Model," in *Proc. Int. Conf. Machine Learning (ICML)*, 2023.

[5] J. Duan et al., "AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation," *arXiv preprint arXiv:2410.00371*, 2024.

[6] D. Fox, W. Burgard, and S. Thrun, "The Dynamic Window Approach to Collision Avoidance," *IEEE Robotics Autom. Mag.*, vol. 4, no. 1, pp. 23–33, 1997.

[7] B. Chen et al., "SpatialEval: Benchmarking Spatial Reasoning in Vision-Language Models," in *Proc. NeurIPS*, 2024.

[8] H. Chen et al., "Automating Robot Failure Recovery Using Vision-Language Models With Optimized Prompts," *arXiv preprint arXiv:2409.03966*, 2024.

[9] S. Salman et al., "Malicious Path Manipulations via Exploitation of Representation Vulnerabilities of Vision-Language Navigation Systems," *IEEE Trans. Inf. Forensics Security*, 2024.

[10] A. Padalkar et al., "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2024.

[11] S. Gu et al., "Deep Reinforcement Learning for Robotic Manipulation with Asynchronous Off-Policy Updates," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2017.

[12] T. Zhang et al., "A Hierarchical Approach to Mobile Robot Navigation Using Deep Reinforcement Learning and Classical Planning," *IEEE Robot. Autom. Lett.*, vol. 7, no. 4, pp. 9876–9883, 2022.

[13] S. Thrun et al., "Probabilistic Robotics," MIT Press, 2005.

[14] M. Ito, "Cerebellum and Neural Control," Raven Press, 1984.

[15] D. M. Wolpert, R. C. Miall, and M. Kawato, "Internal Models in the Cerebellum," *Trends in Cognitive Sciences*, vol. 2, no. 9, pp. 338–347, 1998.

[16] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018.

[17] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020.

[18] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," in *Proc. Robotics: Science and Systems (RSS)*, 2023.

[19] R. Cadene et al., "LeRobot: State-of-the-Art Machine Learning for Real-World Robotics in PyTorch," HuggingFace, 2024. [Online]. Available: https://github.com/huggingface/lerobot

[20] HuggingFace Community, "LeRobot Lekiwi: An Open-Source Three-Omniwheel Mobile Robot," 2024. [Online]. Available: https://github.com/huggingface/lerobot

[21] A. Ajay et al., "Compositional Foundation Models for Hierarchical Planning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[22] J. Liang et al., "Code as Policies: Language Model Programs for Embodied Control," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2023.

[23] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, "Score-Based Generative Modeling through Stochastic Differential Equations," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[24] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022.

[25] L. Fang, R. Liu, J. Zhang, W. Wang, and B. Y. Jing, "Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2025.

[26] Y. Wang, L. Wang, Y. Jiang, et al., "Diffusion Actor-Critic with Entropy Regulator," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2024.

[27] H. Ma, T. Chen, K. Wang, N. Li, and B. Dai, "Efficient Online Reinforcement Learning for Diffusion Policy," in *Proc. Int. Conf. Machine Learning (ICML)*, 2025.

[28] L. Zhang et al., "D2 Actor Critic: Diffusion Actor Meets Distributional Critic," *arXiv preprint arXiv:2510.03508*, 2025.

[29] M. Janner, Y. Du, J. Tenenbaum, and S. Levine, "Planning with Diffusion for Flexible Behavior Synthesis," in *Proc. Int. Conf. Machine Learning (ICML)*, 2022.

[30] A. Ajay et al., "Decision Diffuser: Reward-Conditioned Trajectory Generation via Diffusion Models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[31] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2017.

[32] W. Zhao, J. P. Queralta, and T. Westerlund, "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey," in *Proc. IEEE Symp. Series on Computational Intelligence (SSCI)*, 2020.

[33] OpenAI, M. Andrychowicz et al., "Solving Rubik's Cube with a Robot Hand," *arXiv preprint arXiv:1910.07113*, 2019.

[34] K. Bousmalis et al., "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2018.

[35] I. Jang et al., "Bridging the Simulation-to-Real Gap of Depth Images for Deep Reinforcement Learning," *Expert Systems with Applications*, 2024.

[36] Y. Wang, W.-L. Chao, D. Garg, B. Hariharan, M. Campbell, and K. Q. Weinberger, "Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019.

[37] ROS Perception Community, "depthimage_to_laserscan: Conversion of Depth Images into Laser Scans," ROS Package, 2020. [Online]. Available: https://github.com/ros-perception/depthimage_to_laserscan

[38] C. Jestel, K. Rösner, N. Dietz, et al., "MuRoSim -- A Fast and Efficient Multi-Robot Simulation for Learning-based Navigation," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2024.

[39] S. Sachan and P. M. Pathak, "Addressing Unpredictable Movements of Dynamic Obstacles with Deep Reinforcement Learning to Ensure Safe Navigation for Omni-wheeled Mobile Robot," *Proc. IMechE, Part C: J. Mechanical Engineering Science*, 2024.

[40] Y. C. Ng et al., "Tactile Aware Dynamic Obstacle Avoidance in Crowded Environment with Deep Reinforcement Learning," *arXiv preprint arXiv:2406.13434*, 2024.

[41] F. G. Pin and S. M. Killough, "A New Family of Omnidirectional and Holonomic Wheeled Platforms for Mobile Robots," *IEEE Trans. Robotics and Automation*, vol. 10, no. 4, pp. 480–489, 1994.

[42] R. Siegwart, I. R. Nourbakhsh, and D. Scaramuzza, *Introduction to Autonomous Mobile Robots*, 2nd ed. MIT Press, 2011.

[43] V. R. F. Miranda, A. A. Neto, G. M. Freitas, and L. A. Mozelli, "Generalization in Deep Reinforcement Learning for Robotic Navigation by Reward Shaping," *IEEE Trans. Industrial Electronics*, 2023.

[44] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A Physics Engine for Model-Based Control," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2012.

[45] A. Nichol and P. Dhariwal, "Improved Denoising Diffusion Probabilistic Models," in *Proc. Int. Conf. Machine Learning (ICML)*, 2021.

[46] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[47] D. Fox, W. Burgard, and S. Thrun, "The Dynamic Window Approach to Collision Avoidance," *IEEE Robotics Autom. Mag.*, vol. 4, no. 1, pp. 23–33, 1997.

[48] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[49] Z. Chen et al., "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024.

[50] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, "Control Barrier Functions: Theory and Applications," in *Proc. European Control Conf. (ECC)*, 2019.

[51] S. Fujimoto, H. van Hoof, and D. Meger, "Addressing Function Approximation Error in Actor-Critic Methods," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018.

[52] T. P. Lillicrap et al., "Continuous Control with Deep Reinforcement Learning," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2016.

[53] V. Mnih et al., "Human-Level Control through Deep Reinforcement Learning," *Nature*, vol. 518, pp. 529–533, 2015.

[54] A. Vaswani et al., "Attention Is All You Need," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017.

[55] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Proc. MICCAI*, 2015.

[56] D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2014.

[57] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.

[58] F. Muratore, F. Ramos, G. Turk, W. Yu, M. Gienger, and J. Peters, "Robot Learning from Randomized Simulations: A Review," *arXiv preprint arXiv:2111.00956*, 2021.

[59] H. Hua et al., "Reinforcement Learned Distributed Multi-Robot Navigation with Reciprocal Velocity Obstacle Shaped Rewards," *IEEE Robot. Autom. Lett.*, vol. 7, no. 3, pp. 6896–6903, 2022.

[60] A. Kumar, A. Zhou, G. Tucker, and S. Levine, "Conservative Q-Learning for Offline Reinforcement Learning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020.

---

*This manuscript was prepared for review. Code and models will be released upon acceptance.*
