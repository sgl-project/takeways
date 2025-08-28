# Awesome SGLang

Make [SGLang](https://github.com/sgl-project/sglang) go brrr. 🚀  

[SGLang](https://github.com/sgl-project/sglang) comes with a wide range of configuration parameters. While [the manual](https://docs.sglang.ai/) covers everything in detail, what’s often most useful is a simple, ready-to-use setup that delivers strong performance out of the box—especially when low latency is critical.

The examples in this repo focus on the H100 and B200 GPUs, but our goal is to make this a genuinely community-driven effort. Over time, we hope to see contributions that expand support beyond these platforms. All benchmarks are proudly powered by [genai-bench](https://github.com/sgl-project/genai-bench)➚.

In production deployments, features like EAGLE 3, function calling, JSON mode, reasoning parser, metrics, and crash dump are typically enabled to improve performance, reliability, and observability. The EAGLE 3 training pipeline is powered by [SpecForge](https://github.com/sgl-project/SpecForge)➚, and for Kubernetes orchestration, we recommend [OME](https://github.com/sgl-project/ome)—a production-ready framework designed for scalable LLM serving.

