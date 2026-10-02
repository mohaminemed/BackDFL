# Attack Implementations

This folder provides implementations of established *targeted* backdoor attacks (BAs) for Federated Learning (FL). It also includes established *untargeted* poisoning attacks to support broader evaluation of new defenses, including model poisoning attacks (MPAs) such as Krum, Trim, Gaussian, and OMP, and data poisoning attacks (DPAs) such as label flipping and feature poisoning.

| Attack                      | Reference                               |
| --------------------------- | --------------------------------------- |
| BadNets                     | Gu et al., **IEEE Access'19** [14]      |
| Model Replacement / Scaling | Bagdasaryan et al., **AISTATS'20** [13] |
| DBA                         | Xie et al., **ICLR'20** [52]            |
| Neurotoxin                  | Zhang et al., **ICML'22** [18]          |
| A3FL                        | Zhang et al., **NeurIPS'23** [15]       |
| IBA                        | Nguyen et al., **NeurIPS'23** [16]      |
| LayerPoison                 | Zhuang et al., **[ICLR'24]** [41]    |
| LayerFlip                   | Zhuang et al., **[ICLR'24]** [41]    |

### BadNets

**BadNets** implements a conventional static backdoor attack in which a fixed trigger is inserted into a fraction of local training samples and the poisoned samples are relabeled to a target class. Malicious clients train on the resulting poisoned dataset during the attack window.

### Model Replacement / Scaling

**Model Replacement / Scaling** amplifies a malicious client's locally trained model update before it is propagated to neighboring clients. Following standard backdoor poisoning, the attacker scales the local update by a factor $\lambda$ to increase its influence on the aggregated model. The attack is typically applied in a single round.

### DBA

**DBA (Distributed Backdoor Attack)** distributes the backdoor trigger across multiple malicious clients. Each attacker embeds a distinct trigger pattern into a fraction of its local training samples and relabels the poisoned samples to the same target class. The different trigger components are jointly learned through aggregation, forming the complete backdoor trigger at the global level.

### Neurotoxin

**Neurotoxin** identifies model parameters considered important for the current task based on the magnitude of their gradients relative to their parameter values. The attacker suppresses updates to these critical parameters while training on poisoned data, preserving task-related parameters and concentrating the backdoor update on less important parameters. This is intended to improve backdoor persistence across subsequent training rounds.

### A3FL

**A3FL (Adversarially Adaptive Backdoor Attack)** adaptively optimizes the backdoor trigger against the current model during the attack window. The attacker uses an adversarially hardened surrogate and projected gradient-based optimization to construct a trigger that remains effective against model changes. The optimized trigger is then used for local backdoor training.

### IBA

**IBA (Irreversible Backdoor Attack)** learns a lightweight trigger generator that produces input-dependent perturbations for poisoned samples. During the attack window, the generator is optimized against the current model, and the resulting sample-specific triggers are used for local backdoor training. This allows the attack to employ adaptive triggers rather than a single fixed perturbation.

### LayerPoison

**LayerPoison (Layer-wise Poisoning Attack)** identifies the subset of model layers that are most critical to maintaining the backdoor using Layer Substitution Analysis (LSA). The attacker ranks layers according to their contribution to backdoor success and incrementally constructs a critical layer set $L^*$. The exposed update selectively pulls these layers toward the backdoored model while leaving the remaining layers close to an estimate of the benign population model. This concentrates the malicious update on backdoor-critical layers while reducing unnecessary changes to other parameters.

### LayerFlip

**LayerFlip (Layer-wise Flipping Attack)** reuses the backdoor-critical layer set $L^*$ identified by LSA and targets defenses based on parameter-wise sign agreement. The attacker reverses the update direction on the selected layers before submission. If the defense subsequently reverses updates that disagree with the majority direction, this operation can restore the intended malicious update rather than suppressing it.