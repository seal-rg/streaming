"""Security/prompt-injection-robustness benchmark suite. Fully self-contained:
all non-pip dependencies (RuLES, the Qwen3ForMultiStream architecture,
StruQ/prompt-template data) are vendored under vendor/ and data/.

Covers TensorTrust, Gandalf, Purple, RuLES, StruQ, NESSiE, IFEval.
Model-agnostic: works with any backend implementing the `InferenceBackend`
protocol in `backends.py`.
"""
