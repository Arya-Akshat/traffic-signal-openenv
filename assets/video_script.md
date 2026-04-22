# Traffic Signal OpenEnv: 90-Second Demo Script

**[00:00 - 00:15] The Hook**
*(Visual: Real-world footage of a gridlocked intersection or a busy 2x2 simulation grid)*
"Every day, cities lose millions of dollars to uncoordinated traffic flow. Local sensors are great at managing one intersection, but they’re blind to the city-wide 'spillback' effect that causes gridlock."

**[00:15 - 00:40] The Solution**
*(Visual: ASCII diagram of the 2x2 grid, then the YAML `text_obs` scrolling)*
"Introducing Traffic Signal OpenEnv. We’ve built a hierarchical orchestration benchmark where an LLM acts as the 'Central Controller.' By reading high-fidelity text observations, the LLM sets policy vectors for local agents, managing everything from corridor synchronization to emergency routing."

**[00:40 - 01:10] The Evidence**
*(Visual: Split screen showing 'Central OFF' vs 'Central ON'. The ON side is flowing smoothly, OFF is red with queues.)*
"The results are clear. In our ablation study, Central Oversight improved hard-task performance by a massive 36.2%. It doesn't just respond to traffic—it anticipates it, throttling upstream flow to prevent downstream blockages before they happen."

**[01:10 - 01:30] Call to Action**
*(Visual: Reward curve from `plots/reward_curve.png` and the HF logo)*
"Built with Unsloth and OpenEnv, our benchmark is a playground for Multi-Agent interactions and long-horizon planning. Explore our tasks, run the curriculum, and help us build the next generation of scalable urban oversight."

**[01:30] End**
*(Visual: Project Name and GitHub URL)*
