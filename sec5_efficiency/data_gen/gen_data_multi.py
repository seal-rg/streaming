"""
Multi-Agent Collaborative Reasoning System
Clean implementation with comprehensive tracking
"""

import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import numpy as np
import shared_cache
import torch
import transformers
from datasets import load_dataset
from external_sample import sample_tokens_like_transformers
from multi_agent_complete_system import MultiAgentFormatter, create_coordinator_team, create_peer_team

# ===== DATA STRUCTURES =====


class TokenAlignedStepData(NamedTuple):
    step_num: int
    content_text: str
    token_count: int


class TokenAlignedAgentData(NamedTuple):
    completed_steps: list[TokenAlignedStepData]
    current_step_text: str
    total_tokens: int


class TokenAlignedReasoningState(NamedTuple):
    history: Sequence[int]
    current_step_tokens_by_worker: Sequence[Sequence[int]]
    finished: bool
    agent_data: dict[str, TokenAlignedAgentData]
    early_terminated: bool
    termination_details: dict


# ===== VERIFICATION SYSTEM =====


class VerificationHistory:
    """Manages early termination decisions based on answer consensus"""

    def __init__(self, required_checks: int = 3):
        self.required_checks = required_checks
        self.recent_answers = []
        self.termination_info = None
        self.check_history = []

    def check_termination(self, answers: list[str], true_answer: str, step: int) -> tuple[bool, str]:
        """Check if should terminate early. Returns (should_terminate, reason)"""
        correct_count = sum(1 for ans in answers if ans and self._match(ans, true_answer))
        consensus = self._find_consensus(answers)

        # Record check
        self.check_history.append({"step": step, "answers": answers, "consensus": consensus, "correct_count": correct_count})

        # Immediate termination if majority correct
        if correct_count >= max(1, len(answers) // 2):
            self.termination_info = {
                "type": "correct_immediate",
                "step_number": step,
                "final_answers": answers,
                "consensus_answer": consensus,
                "true_answer": true_answer,
                "accuracy": correct_count / len(answers),
                "check_history": self.check_history,
            }
            return True, "correct_immediate"

        # Check for convergence
        if consensus:
            self.recent_answers.append(consensus)
            if len(self.recent_answers) > self.required_checks:
                self.recent_answers = self.recent_answers[-self.required_checks :]

            if len(self.recent_answers) >= self.required_checks:
                if all(self._match(ans, self.recent_answers[0]) for ans in self.recent_answers):
                    self.termination_info = {
                        "type": "converged",
                        "step_number": step,
                        "final_answers": answers,
                        "converged_answer": consensus,
                        "true_answer": true_answer,
                        "matches_true_answer": self._match(consensus, true_answer),
                        "consecutive_answers": self.recent_answers.copy(),
                        "check_history": self.check_history,
                    }
                    return True, "converged"

        return False, ""

    def _find_consensus(self, answers: list[str]) -> str | None:
        """Find most common answer"""
        valid = [ans for ans in answers if ans and ans.strip()]
        if not valid:
            return None
        counts = Counter(self._normalize(ans) for ans in valid)
        most_common = counts.most_common(1)[0][0]
        return next(ans for ans in valid if self._normalize(ans) == most_common)

    def _normalize(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""
        answer = answer.strip().lower()
        answer = re.sub(r"[\\$(),.]", "", answer)
        return answer.strip()

    def _match(self, ans1: str, ans2: str) -> bool:
        """Check if answers match"""
        norm1, norm2 = self._normalize(ans1), self._normalize(ans2)
        if not norm1 or not norm2:
            return False
        if norm1 == norm2:
            return True
        try:
            return abs(float(norm1) - float(norm2)) < 1e-6
        except:
            return False


# ===== ANSWER EXTRACTION =====


def extract_answer(text: str) -> str:
    """Extract final answer from text"""
    if not text:
        return ""

    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"(?:final answer|answer|therefore.*answer) is\s*([^.,]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return re.sub(r"[\\$]", "", matches[-1].strip()).rstrip(".,")
    return ""


def evaluate_answers(responses: list[str], true_answer: str) -> list[str]:
    """Extract and return agent answers"""
    answers = []
    for response in responses:
        answer = extract_answer(response)
        if not answer and response.strip():
            numbers = re.findall(r"-?\d+\.?\d*", response)
            answer = numbers[-1] if numbers else ""
        answers.append(answer)
    return answers


# ===== CONFIDENCE CHECK =====


def generate_confidence_responses(
    problem: str,
    agent_data: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    fmt: MultiAgentFormatter,
    history: list[int],
    current_tokens: list[list[int]],
    max_tokens: int = 32,
) -> list[str]:
    """Generate confidence check responses"""
    device = next(model.parameters()).device
    responses = []

    prompt = "The final answer is \\boxed{"
    problem_ids = tokenizer.encode(fmt.get_full_prompt(problem), add_special_tokens=False)

    for idx, agent_name in enumerate([a.name for a in fmt.agents]):
        try:
            input_ids = problem_ids + history
            input_ids += tokenizer.encode(fmt.work_in_progress_others + fmt.step_separator, add_special_tokens=False)
            if idx < len(current_tokens):
                input_ids += current_tokens[idx]
            input_ids += tokenizer.encode(prompt, add_special_tokens=False)

            input_tensor = torch.tensor([input_ids], device=device, dtype=torch.int64)

            with torch.inference_mode():
                output = model.generate(
                    input_tensor, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True
                )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(response)
        except Exception as e:
            print(f"Error in confidence check for {agent_name}: {e}")
            responses.append(agent_data.get(agent_name, {}).get("all_text", ""))

    return responses


# ===== TOKEN PROCESSING =====


def get_logits_processor(model: transformers.PreTrainedModel, forbidden_tokens: Sequence[int]):
    """Get logits processor"""
    config, kwargs = model._prepare_generation_config(model.generation_config)
    model._prepare_special_tokens(config)
    device = next(model.parameters()).device
    return model._get_logits_processor(
        generation_config=config,
        input_ids_seq_length=0,
        encoder_input_ids=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=transformers.LogitsProcessorList(
            [transformers.generation.logits_process.SuppressTokensLogitsProcessor(forbidden_tokens, device=device)]
        ),
        device=device,
        model_kwargs=kwargs,
    )


def extract_content_tokens(
    full_tokens: list[int], fmt: MultiAgentFormatter, tokenizer: transformers.PreTrainedTokenizer, agent_name: str
) -> list[int]:
    """Extract content tokens without prefix"""
    if not full_tokens:
        return []

    text = tokenizer.decode(full_tokens, skip_special_tokens=True)
    patterns = [
        rf"\*\*{re.escape(agent_name)}\s*\[[^\]]+\]\*\*:\s*",
        rf"\*\*{re.escape(agent_name)}\*\*:\s*",
        rf"{re.escape(agent_name)}:\s*",
    ]

    prefix_end = 0
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prefix_end = match.end()
            break

    if prefix_end == 0:
        return full_tokens

    prefix_tokens = tokenizer.encode(text[:prefix_end], add_special_tokens=False)
    return full_tokens[len(prefix_tokens) :] if len(prefix_tokens) < len(full_tokens) else []


# ===== CACHE MANAGEMENT =====


def create_cache_structure(num_agents: int, config) -> tuple[list[list], shared_cache.SharedCacheManager]:
    """Create cache structure for agents"""
    common = shared_cache.CacheBlock(config=config)
    step_header = shared_cache.CacheBlock(config=config)
    own_header = shared_cache.CacheBlock(config=config)
    agent_caches = [shared_cache.CacheBlock(config=config) for _ in range(num_agents)]

    structure = []
    for i in range(num_agents):
        path = [common, step_header]
        path.extend([agent_caches[j] for j in range(num_agents) if j != i])
        path.extend([own_header, agent_caches[i]])
        structure.append(path)

    return structure, shared_cache.SharedCacheManager(cache_structure=structure)


# ===== MAIN REASONING =====


def generate_reasoning(
    problem: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    fmt: MultiAgentFormatter,
    max_steps: int,
    true_answer: str = "",
    check_interval: int = 1024,
    required_checks: int = 3,
) -> TokenAlignedReasoningState:
    """Generate multi-agent reasoning"""

    logits_processor = get_logits_processor(model, fmt.forbidden_token_ix)
    device = next(model.parameters()).device
    tok_kwargs = dict(return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)

    agent_names = [a.name for a in fmt.agents]
    num_agents = len(agent_names)

    cache_structure, cache_mgr = create_cache_structure(num_agents, model.config)

    agent_data = {name: {"completed_steps": [], "current_tokens": [], "total_tokens": 0, "all_text": ""} for name in agent_names}

    verifier = VerificationHistory(required_checks)

    # Prefill cache
    with torch.inference_mode():
        model(**tokenizer(fmt.get_full_prompt(problem), **tok_kwargs).to(device), use_cache=True, past_key_values=cache_structure[0][0])
        model(**tokenizer(fmt.current_step_header, **tok_kwargs).to(device), use_cache=True, past_key_values=cache_structure[0][1])
        model(**tokenizer(fmt.current_worker_header, **tok_kwargs).to(device), use_cache=True, past_key_values=cache_structure[0][-2])

    prompts = [fmt.get_step_prefix(a.name, 1) + f"Hi, I'm {a.name}" for a in fmt.agents]
    step_indices = [1] * num_agents
    current_tokens = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    history = []
    finished = False
    early_term = False
    term_details = {}
    tokens_since_check = 0

    for name, tokens in zip(agent_names, current_tokens):
        agent_data[name]["current_tokens"] = tokens.copy()

    next_inputs = tokenizer(prompts, **tok_kwargs).to(device)

    # Main loop
    for step in range(max_steps):
        if finished or early_term:
            break

        with torch.inference_mode():
            logits = model(**cache_mgr.get_input_kwargs(**next_inputs)).logits[..., -1, :]
            logits = logits_processor(next_inputs["input_ids"], logits)
            new_tokens = sample_tokens_like_transformers(
                logits=logits,
                input_ids=cache_mgr.get_input_kwargs(**next_inputs)["input_ids"],
                # top_p=0.95,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
            )
        # for gpt-oss-20b,  We recommend sampling with temperature=1.0 and top_p=1.0
        # for Qwen-3-32b, Temperature=0.6, TopP=0.95, TopK=20, and MinP=0

        next_token_list = new_tokens.unsqueeze(-1).tolist()
        tokens_since_check += len(new_tokens)

        # Early termination check
        if true_answer and step > 1024 and tokens_since_check >= check_interval:
            try:
                responses = generate_confidence_responses(problem, agent_data, tokenizer, model, fmt, history, current_tokens, 32)
                answers = evaluate_answers(responses, true_answer)
                should_term, reason = verifier.check_termination(answers, true_answer, step)

                if should_term:
                    print(f"Early termination at step {step}: {reason}")
                    term_details = verifier.termination_info
                    early_term = True
                    break
            except Exception as e:
                print(f"Check error: {e}")

            tokens_since_check = 0

        # Process tokens
        for idx, (name, tokens, token) in enumerate(zip(agent_names, current_tokens, new_tokens.tolist())):
            tokens.append(token)
            agent_data[name]["current_tokens"].append(token)

            if fmt.is_end_of_step(tokens):
                if fmt.should_finish_reasoning(tokens):
                    finished = True

                content = extract_content_tokens(agent_data[name]["current_tokens"], fmt, tokenizer, name)
                text = tokenizer.decode(content, skip_special_tokens=True).strip()
                agent_data[name]["all_text"] += " " + text

                agent_data[name]["completed_steps"].append(
                    TokenAlignedStepData(step_num=step_indices[idx], content_text=text, token_count=len(content))
                )
                agent_data[name]["total_tokens"] += len(agent_data[name]["current_tokens"])

                step_indices[idx] += 1
                history.extend(tokens)
                tokens.clear()

                start = fmt.get_step_prefix(name, step_indices[idx])
                start_tokens = tokenizer.encode(start, add_special_tokens=False)
                tokens.extend(start_tokens)
                agent_data[name]["current_tokens"] = start_tokens.copy()

                cache_structure[0][0].append_from(cache_structure[idx][-1])
                cache_structure[idx][-1].clear()
                next_token_list[idx] = [token] + tokens

        next_inputs = tokenizer.pad(dict(input_ids=next_token_list), padding_side="left", return_tensors="pt").to(device)

    # Build final state
    final_data = {}
    for name in agent_names:
        curr_text = tokenizer.decode(agent_data[name]["current_tokens"], skip_special_tokens=True)
        final_data[name] = TokenAlignedAgentData(
            completed_steps=agent_data[name]["completed_steps"],
            current_step_text=curr_text,
            total_tokens=agent_data[name]["total_tokens"] + len(agent_data[name]["current_tokens"]),
        )

    return TokenAlignedReasoningState(
        history=history,
        current_step_tokens_by_worker=deepcopy(current_tokens),
        finished=finished,
        agent_data=final_data,
        early_terminated=early_term,
        termination_details=term_details,
    )


# ===== FINISHER =====


@torch.inference_mode()
def generate_finisher(
    problem: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    fmt: MultiAgentFormatter,
    state: TokenAlignedReasoningState,
    agent_name: str,
    max_tokens: int = 512,
) -> str:
    """Generate finisher response"""
    problem_ids = tokenizer.encode(fmt.get_full_prompt(problem), add_special_tokens=False)
    agent_names = list(state.agent_data.keys())

    if agent_name not in agent_names:
        return ""

    idx = agent_names.index(agent_name)
    output = problem_ids + list(state.history)
    output += tokenizer.encode(fmt.work_in_progress_others + fmt.step_separator, add_special_tokens=False)

    if idx < len(state.current_step_tokens_by_worker):
        output += state.current_step_tokens_by_worker[idx]

    output += tokenizer.encode(fmt.pivot_message + fmt.step_separator, add_special_tokens=False)
    response = tokenizer.decode(output)

    if max_tokens > 0 and fmt.get_final_answer(response) is None:
        device = next(model.parameters()).device
        suffix = fmt.step_separator + "The final answer is \\boxed{"
        input_ids = torch.tensor([tokenizer.encode(response + suffix, add_special_tokens=False)], device=device, dtype=torch.int64)

        output = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response


# ===== PROBLEM SOLVING =====


def solve_problem(
    problem_data: dict,
    problem_id: int,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    fmt: MultiAgentFormatter,
    max_steps: int = 1024,
    finisher_tokens: int = 512,
    check_interval: int = 1024,
    required_checks: int = 3,
) -> dict:
    """Solve single problem with comprehensive tracking"""
    problem = problem_data.get("question", "")
    true_answer = extract_answer(problem_data.get("solution", ""))
    if not true_answer:
        true_answer = problem_data.get("solution", "")

    state = generate_reasoning(
        problem=problem,
        model=model,
        tokenizer=tokenizer,
        fmt=fmt,
        max_steps=max_steps,
        true_answer=true_answer,
        check_interval=check_interval,
        required_checks=required_checks,
    )

    # Process each agent with detailed tracking
    agent_results = {}
    total_reasoning_tokens = 0
    total_finisher_tokens = 0

    for name, data in state.agent_data.items():
        # Collect reasoning text as list (step by step)
        reasoning_text_list = [s.content_text for s in data.completed_steps]
        reasoning_text_list.append(data.current_step_text)

        all_text = " ".join(reasoning_text_list)
        reasoning_tokens = data.total_tokens

        # Generate finisher if needed
        finisher_response = ""
        finisher_token_count = 0
        if not state.finished and not state.early_terminated:
            finisher_response = generate_finisher(problem, model, tokenizer, fmt, state, name, finisher_tokens)
            finisher_token_count = len(tokenizer.encode(finisher_response, add_special_tokens=False))
            all_text += " " + finisher_response

        answer = extract_answer(all_text)
        is_correct = answer and true_answer and VerificationHistory()._match(answer, true_answer)

        total_reasoning_tokens += reasoning_tokens
        total_finisher_tokens += finisher_token_count

        agent_results[name] = {
            "final_answer": answer,
            "is_correct": is_correct,
            "reasoning_text": reasoning_text_list,  # Step-by-step text list
            "finisher_response": finisher_response,
            "token_stats": {
                "reasoning_tokens": reasoning_tokens,
                "finisher_tokens": finisher_token_count,
                "total_tokens": reasoning_tokens + finisher_token_count,
            },
        }

    # Determine termination reason
    if state.early_terminated:
        term_type = state.termination_details.get("type", "unknown")
        if term_type == "correct_immediate":
            termination_reason = "early_termination_correct"
        elif term_type == "converged":
            termination_reason = "early_termination_converged"
        else:
            termination_reason = "early_termination"
    elif state.finished:
        termination_reason = "natural_finish"
    else:
        termination_reason = "max_steps_reached"

    return {
        "problem_id": problem_id,
        "problem": problem,
        "ori_problem": problem_data,
        "true_answer": true_answer,
        "agent_results": agent_results,
        "any_correct": any(r["is_correct"] for r in agent_results.values()),
        "all_correct": all(r["is_correct"] for r in agent_results.values()),
        "token_stats": {
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_finisher_tokens": total_finisher_tokens,
            "total_tokens": total_reasoning_tokens + total_finisher_tokens,
        },
        "finished_naturally": state.finished,
        "early_terminated": state.early_terminated,
        "termination_info": {
            "termination_reason": termination_reason,
            "max_steps_reached": not state.finished and not state.early_terminated,
            "details": state.termination_details,
        },
    }


# ===== DATASET SOLVING =====


def solve_dataset(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: list,
    output_dir: str,
    max_steps: int = 1024,
    seed: int = 42,
    finisher_max_new_tokens: int = 512,
    check_interval: int = 1024,
    required_checks: int = 3,
    coordinator_name: str = "Coordinator",
    worker_names: list[str] = None,
    use_coordinator: bool = True,
):
    """Solve dataset with comprehensive statistics"""
    fix_seed(seed)

    if worker_names is None:
        worker_names = ["Alice", "Bob"]

    fmt = (
        create_coordinator_team(coordinator_name, worker_names, tokenizer) if use_coordinator else create_peer_team(worker_names, tokenizer)
    )

    print(f"Mode: {'Coordinator' if use_coordinator else 'Peer'}")
    print(f"Agents: {[a.name for a in fmt.agents]}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_problems": len(dataset),
        "multi_agent_config": {
            "use_coordinator": use_coordinator,
            "coordinator_name": coordinator_name if use_coordinator else None,
            "worker_names": worker_names,
            "total_agents": len(fmt.agents),
        },
        "any_correct": 0,
        "all_correct": 0,
        "early_terminated": 0,
        "finished_naturally": 0,
        "max_steps_reached": 0,
        "termination_reasons": {
            "early_termination_correct": 0,
            "early_termination_converged": 0,
            "natural_finish": 0,
            "max_steps_reached": 0,
        },
    }

    for i, prob in enumerate(dataset):
        print(f"\nProcessing problem {i + 1}/{len(dataset)}")

        try:
            result = solve_problem(
                problem_data=prob,
                problem_id=i,
                model=model,
                tokenizer=tokenizer,
                fmt=fmt,
                max_steps=max_steps,
                finisher_tokens=finisher_max_new_tokens,
                check_interval=check_interval,
                required_checks=required_checks,
            )

            # Update statistics
            if result["any_correct"]:
                stats["any_correct"] += 1
            if result["all_correct"]:
                stats["all_correct"] += 1
            if result["early_terminated"]:
                stats["early_terminated"] += 1
            if result["finished_naturally"]:
                stats["finished_naturally"] += 1

            term_reason = result["termination_info"]["termination_reason"]
            stats["termination_reasons"][term_reason] = stats["termination_reasons"].get(term_reason, 0) + 1

            # Print result
            term_status = f" ({term_reason})"
            print(f"Problem {i + 1} completed{term_status} - True answer: {result['true_answer']}")

            for agent_name, agent_result in result["agent_results"].items():
                status = "✓" if agent_result["is_correct"] else "✗"
                tokens = agent_result["token_stats"]["total_tokens"]
                print(f"  {agent_name}: {agent_result['final_answer']} ({status}) - {tokens} tokens")

            print(
                f"  Total tokens: {result['token_stats']['total_tokens']} "
                f"(reasoning: {result['token_stats']['total_reasoning_tokens']}, "
                f"finisher: {result['token_stats']['total_finisher_tokens']})"
            )

            if result["early_terminated"]:
                term_details = result["termination_info"]["details"]
                print(f"  Termination step: {term_details.get('step_number', 'N/A')}")
                if term_details.get("type") == "correct_immediate":
                    print(f"    Accuracy: {term_details.get('accuracy', 0):.1%}")
                elif term_details.get("type") == "converged":
                    print(
                        f"    Converged to: {term_details.get('converged_answer', '')} "
                        f"({'correct' if term_details.get('matches_true_answer', False) else 'incorrect'})"
                    )

            # Save result
            with open(output_path / f"problem_{i:04d}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    # Final statistics
    stats["accuracy_any"] = stats["any_correct"] / stats["total_problems"]
    stats["accuracy_all"] = stats["all_correct"] / stats["total_problems"]
    stats["early_termination_rate"] = stats["early_terminated"] / stats["total_problems"]

    with open(output_path / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Final Statistics ===")
    print(f"Total: {stats['total_problems']}")
    print(f"Any correct: {stats['any_correct']} ({stats['accuracy_any']:.1%})")
    print(f"All correct: {stats['all_correct']} ({stats['accuracy_all']:.1%})")
    print("\nTermination reasons:")
    for reason, count in stats["termination_reasons"].items():
        if count > 0:
            print(f"  {reason}: {count}")
    print(f"\nResults saved to {output_dir}")


def fix_seed(seed: int):
    """Fix random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


# ===== MAIN =====


def main():
    """Main function"""
    # ${DATA_ROOT}/new_moe/download_models/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
    # "${MODELS_ROOT}/Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
    MODEL_PATH = "${MODELS_ROOT}/Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
    # "${DATA_ROOT}/new_moe/download_models/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    # "${MODELS_ROOT}/Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"

    import argparse

    parser = argparse.ArgumentParser(description="Multi-agent reasoning system")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=16384)
    parser.add_argument("--finisher_tokens", type=int, default=512)
    parser.add_argument("--check_interval", type=int, default=1024)
    parser.add_argument("--required_checks", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto")
    # GAIR/LIMO. simplescaling/s1K-1.1
    print("Loading dataset...")
    dataset = list(load_dataset("simplescaling/s1K-1.1", split="train"))[args.start : args.end]
    print(f"Loaded {len(dataset)} problems")

    solve_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        finisher_max_new_tokens=args.finisher_tokens,
        check_interval=args.check_interval,
        required_checks=args.required_checks,
        coordinator_name="Supervisor",
        worker_names=["Alice", "Bob"],
        use_coordinator=False,
    )


if __name__ == "__main__":
    main()
