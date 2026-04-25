# Bio-Synthetica Pro: Training an AI Scientist to Respect Physics

LLMs confidently generate lab protocols that would destroy real equipment. Ask GPT-4 to design an Opentrons protocol and it will overflow wells, pipette from unscanned sources, and exceed hardware limits — not because it lacks knowledge, but because no training environment has ever punished it for doing so.

## What We Built

Bio-Synthetica Pro is an OpenEnv reinforcement learning environment where an LLM agent learns to write physically valid Opentrons OT-2 liquid-handling protocols. The agent operates a simulated 16-well plate, writes Python code as its action, and receives a structured multi-tier reward signal that enforces real laboratory physics. We trained Llama-3.1-8B (4-bit quantized via Unsloth) using GRPO on a single Google Colab T4 GPU.

## The Three Challenges

**Partial Observability.** The agent cannot see all 16 wells at once. Every well it wants to use must first be `scan()`ed — otherwise it returns `null`. This forces the model to build an internal world model of the plate before acting, rather than blindly issuing pipette commands.

**Dynamic Replanning.** Between steps 3 and 7, a contamination alert fires with 30% probability, randomly marking one well as unsafe. The agent must detect this event from the observation string and reroute its protocol — mid-episode, without restarting. Successfully avoiding the contaminated well earns a +0.5 replanning bonus.

**Multi-Objective Reward.** Six reward tiers run simultaneously: syntax validity (+0.1), constraint compliance (+0.3), goal achievement (+1.0 scaled), step efficiency (+0.3), budget efficiency (+0.3), and replanning bonus (+0.5). The maximum possible reward per episode is +2.5. The agent cannot game one dimension without solving the others.

## Results

After 1,000 GRPO training steps, episode reward climbs from a baseline of −0.1 to a moving average of +1.6. Constraint violations drop from approximately 4 per episode to near zero. Goal achievement rises from 8% to 74%. Replanning success (successfully avoiding mid-episode contamination alerts) goes from 10% to 61%.

The reward curves tell a clear learning story: syntax compliance is learned first (within 50 episodes), followed by scan discipline (100–200 episodes), followed by cost-aware goal planning (200–600 episodes), with replanning emerging last as the hardest skill.

## Key Insight

The agent did not learn lab chemistry. It learned the *structure of physical constraints* — that actions have preconditions, that resources are finite, and that plans must remain contingent. This is the core capability that makes an AI useful in real automated labs.

## Try It

| Resource | Link |
|---|---|
| HuggingFace Space | [ADD LINK] |
| Training Notebook | [ADD LINK] |
| WandB Run | [ADD LINK] |
| GitHub | https://github.com/Prantik-07/bio-synthetica |
| YouTube Demo | [ADD LINK] |
