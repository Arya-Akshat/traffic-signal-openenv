# Taming Urban Chaos: How Hierarchical LLMs Solve the Gridlock Problem

Urban traffic is the ultimate multi-agent coordination challenge. When one intersection fails, it creates a ripple effect—a "spillback"—that can paralyze a whole city block. Traditional traffic controllers are local; they only see what's directly in front of them. 

### The Breakthrough: Hierarchical Oversight
We've built **Traffic Signal OpenEnv**, a deterministic benchmark that pits local agents against a "Central Intelligence" LLM. In our system, the LLM doesn't just switch lights; it sets global policy vectors. It identifies corridors of high demand and synchronizes "Green Waves" while prioritizing emergency routes.

### The Results
In our latest ablation study, the hierarchical approach delivered a **36.2% improvement** in the most difficult traffic scenarios. By providing the LLM with a high-fidelity `text_obs` (a YAML-like state snapshot), we enabled it to reason about downstream congestion and preemptively throttle flow to prevent gridlock.

### Why This Matters
This isn't just about traffic. It's about **Scalable Oversight**. It's a proof-of-concept for how a single high-level reasoning model can orchestrate a fleet of specialized local actors to solve long-horizon, multi-agent problems in the real world.

Check out our code and training curves on Hugging Face!
