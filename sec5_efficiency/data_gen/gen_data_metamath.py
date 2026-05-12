"""
Multi-Agent Collaborative Reasoning System
Adapted for MetaMath dataset
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


def extract_answer_from_metamath_response(response: str) -> str:
    """
    Extract final answer from MetaMath response format.
    MetaMath responses typically end with "The answer is: <answer>" or use \\boxed{}.
    """
    if not response:
        return ""

    # Pattern 1: "The answer is: <answer>" (most common in MetaMath)
    answer_is_pattern = r"[Tt]he answer is:?\s*(.+?)(?:\.|$)"
    matches = re.findall(answer_is_pattern, response)
    if matches:
        answer = matches[-1].strip()
        # Clean up common formatting
        answer = re.sub(r"[\\$]", "", answer)
        return answer.rstrip(".,")

    # Pattern 2: \\boxed{<answer>}
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(boxed_pattern, response)
    if matches:
        return re.sub(r"[\\$]", "", matches[-1].strip()).rstrip(".,")

    # Pattern 3: "#### <answer>" (GSM8K format)
    hash_pattern = r"####\s*(.+?)(?:\n|$)"
    matches = re.findall(hash_pattern, response)
    if matches:
        return matches[-1].strip()

    return ""


def extract_answer(text: str) -> str:
    """Extract final answer from text (for model-generated responses)"""
    if not text:
        return ""

    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"[Tt]he answer is:?\s*(.+?)(?:\.|$)",
        r"####\s*(.+?)(?:\n|$)",
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


# ===== METAMATH DATASET PROCESSING =====


def load_metamath_dataset(
    dataset_name: str = "meta-math/MetaMathQA",
    split: str = "train",
    start: int = 0,
    end: int = 50,
    problem_types: list[str] | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    Load and preprocess MetaMath dataset.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        start: Start index
        end: End index
        problem_types: Filter by problem types (e.g., ["MATH_AnsAug", "GSM_Rephrased"])
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        List of problem dictionaries with standardized format
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    # Convert to list for easier manipulation
    data = list(dataset)

    # Filter by problem types if specified
    if problem_types:
        data = [d for d in data if d.get("type") in problem_types]
        print(f"Filtered to {len(data)} problems of types: {problem_types}")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    # Slice dataset
    data = data[start:end]
    print(f"Selected {len(data)} problems (index {start} to {end})")

    # Standardize format to match the expected structure
    standardized_data = []
    for item in data:
        # Extract the true answer from the response
        true_answer = extract_answer_from_metamath_response(item.get("response", ""))

        standardized_item = {
            "question": item.get("query", item.get("original_question", "")),
            "original_question": item.get("original_question", ""),
            "solution": item.get("response", ""),
            "true_answer": true_answer,
            "type": item.get("type", "unknown"),
            # Keep original data for reference
            "_original": item,
        }
        standardized_data.append(standardized_item)

    return standardized_data


def get_true_answer_from_metamath(problem_data: dict) -> str:
    """
    Get the true answer from a MetaMath problem.

    Args:
        problem_data: Problem dictionary

    Returns:
        Extracted answer string
    """
    # First check if we already extracted it
    if problem_data.get("true_answer"):
        return problem_data["true_answer"]

    # Try to extract from solution/response
    solution = problem_data.get("solution", problem_data.get("response", ""))
    return extract_answer_from_metamath_response(solution)


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
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
            )

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

    # Get true answer (adapted for MetaMath)
    true_answer = get_true_answer_from_metamath(problem_data)

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
            "reasoning_text": reasoning_text_list,
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
        "problem_type": problem_data.get("type", "unknown"),
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

    # Track statistics by problem type
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
        "by_problem_type": {},  # MetaMath-specific: track stats by problem type
    }

    for i, prob in enumerate(dataset):
        problem_type = prob.get("type", "unknown")
        print(f"\nProcessing problem {i + 1}/{len(dataset)} (Type: {problem_type})")

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

            # Update overall statistics
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

            # Update per-type statistics
            if problem_type not in stats["by_problem_type"]:
                stats["by_problem_type"][problem_type] = {"total": 0, "any_correct": 0, "all_correct": 0}
            stats["by_problem_type"][problem_type]["total"] += 1
            if result["any_correct"]:
                stats["by_problem_type"][problem_type]["any_correct"] += 1
            if result["all_correct"]:
                stats["by_problem_type"][problem_type]["all_correct"] += 1

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
    stats["accuracy_any"] = stats["any_correct"] / stats["total_problems"] if stats["total_problems"] > 0 else 0
    stats["accuracy_all"] = stats["all_correct"] / stats["total_problems"] if stats["total_problems"] > 0 else 0
    stats["early_termination_rate"] = stats["early_terminated"] / stats["total_problems"] if stats["total_problems"] > 0 else 0

    # Calculate per-type accuracy
    for ptype, pstats in stats["by_problem_type"].items():
        pstats["accuracy_any"] = pstats["any_correct"] / pstats["total"] if pstats["total"] > 0 else 0
        pstats["accuracy_all"] = pstats["all_correct"] / pstats["total"] if pstats["total"] > 0 else 0

    with open(output_path / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Final Statistics ===")
    print(f"Total: {stats['total_problems']}")
    print(f"Any correct: {stats['any_correct']} ({stats['accuracy_any']:.1%})")
    print(f"All correct: {stats['all_correct']} ({stats['accuracy_all']:.1%})")

    print("\nBy Problem Type:")
    for ptype, pstats in stats["by_problem_type"].items():
        print(f"  {ptype}: {pstats['any_correct']}/{pstats['total']} ({pstats['accuracy_any']:.1%})")

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
    MODEL_PATH = "${MODELS_ROOT}/Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

    import argparse

    parser = argparse.ArgumentParser(description="Multi-agent reasoning system for MetaMath")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="./results_metamath")
    parser.add_argument("--dataset", type=str, default="meta-math/MetaMathQA", help="HuggingFace dataset name")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=16384)
    parser.add_argument("--finisher_tokens", type=int, default=512)
    parser.add_argument("--check_interval", type=int, default=1024)
    parser.add_argument("--required_checks", type=int, default=3)
    parser.add_argument(
        "--problem_types", type=str, nargs="*", default=None, help="Filter by problem types (e.g., MATH_AnsAug GSM_Rephrased)"
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_coordinator", action="store_true", help="Use coordinator mode instead of peer mode")
    parser.add_argument("--coordinator_name", type=str, default="Supervisor")
    parser.add_argument("--worker_names", type=str, nargs="+", default=["Alice", "Bob"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading model from {args.model_path}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto"
    )

    # Load MetaMath dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_metamath_dataset(
        dataset_name=args.dataset,
        split="train",
        start=args.start,
        end=args.end,
        problem_types=args.problem_types,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} problems")

    # Show sample problem types
    types_count = {}
    for p in dataset:
        t = p.get("type", "unknown")
        types_count[t] = types_count.get(t, 0) + 1
    print(f"Problem type distribution: {types_count}")

    solve_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        finisher_max_new_tokens=args.finisher_tokens,
        check_interval=args.check_interval,
        required_checks=args.required_checks,
        coordinator_name=args.coordinator_name,
        worker_names=args.worker_names,
        use_coordinator=args.use_coordinator,
    )


if __name__ == "__main__":
    main()
