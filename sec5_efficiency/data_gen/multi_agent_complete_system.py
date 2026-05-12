"""
Complete Multi-Agent Collaborative Reasoning System
A comprehensive framework for coordinated AI agent problem-solving
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

import transformers

T = TypeVar("T")


class AgentRole(Enum):
    """Types of agent roles"""

    COORDINATOR = "coordinator"
    WORKER = "worker"


class InterventionType(Enum):
    """Types of coordinator interventions"""

    ERROR = "ERR"
    INTERRUPT = "INT"
    REDIRECT = "REDIR"
    BUILD = "BUILD"
    CHECK = "CHECK"
    FINAL = "FINAL"


@dataclass
class AgentConfig:
    """Agent configuration"""

    name: str
    role: AgentRole
    specialization: str | None = None


class MultiAgentFormatter:
    """Multi-agent collaborative reasoning formatter"""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        agents: list[AgentConfig] = None,
        extract_result: Callable[[str], T] = float,
        pass_system_prompt_as_user_message: bool = True,
    ):

        # Default configuration: 1 coordinator + 3 workers
        if agents is None:
            agents = [
                AgentConfig("Coordinator", AgentRole.COORDINATOR),
                AgentConfig("Alice", AgentRole.WORKER),
                AgentConfig("Bob", AgentRole.WORKER),
                AgentConfig("Charlie", AgentRole.WORKER),
            ]

        self._validate_agents(agents)

        # Core configuration
        self.tokenizer = tokenizer
        self.agents = agents
        self.extract_result = extract_result
        self.pass_system_prompt_as_user_message = pass_system_prompt_as_user_message

        # Formatting constants
        self.step_separator = "\n\n"
        self.history_header = "### Past steps"
        self.work_in_progress_others = "### Work in progress (others)"
        self.work_in_progress_self = "### Work in progress (own)"
        self.pivot_message = "<...>"
        self.begin_of_reasoning = "<think>"
        self.final_answer_example = "\\boxed{answer here}"
        self.end_of_step_chars = [".", "?", "!", "。", "۔", "؟", "।", "॥", "…", "‽", "།", "᠃", "։", "჻", "¶", "❧"]  # before SEP
        self.s1_collab_message = "Quick check: am I doing redundant work? (yes/no): "
        self.s1_finisher_suffix = (
            f"{self.step_separator}Wait, given the limited time, I have to give an answer right now. "
            "Considering all my previous attempts, I have to conclude that the final answer is \\boxed{"
        )
        self.current_step_header = self.step_separator + self.work_in_progress_others + self.step_separator
        self.current_worker_header = self.pivot_message + self.step_separator + self.work_in_progress_self + self.step_separator

        # Agent organization
        self.coordinator = self._find_coordinator()
        self.workers = self._find_workers()

        # Setup
        self._setup_tokenizer()
        self.system_prompt = self._build_system_prompt()

    def _validate_agents(self, agents: list[AgentConfig]):
        """Validate agent configuration"""
        if not agents:
            raise ValueError("At least one agent required")

        names = [agent.name for agent in agents]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")

        coordinators = [a for a in agents if a.role == AgentRole.COORDINATOR]
        if len(coordinators) > 1:
            raise ValueError("Maximum one coordinator allowed")

    def _find_coordinator(self) -> AgentConfig | None:
        """Find coordinator agent"""
        coordinators = [a for a in self.agents if a.role == AgentRole.COORDINATOR]
        return coordinators[0] if coordinators else None

    def _find_workers(self) -> list[AgentConfig]:
        """Find worker agents"""
        return [a for a in self.agents if a.role == AgentRole.WORKER]

    def has_coordinator(self) -> bool:
        """Check if system has coordinator"""
        return self.coordinator is not None

    def _setup_tokenizer(self):
        """Setup tokenizer configurations"""
        forbidden_tokens = ["#", self.tokenizer.bos_token, self.tokenizer.eos_token, "</think>"]
        self.forbidden_token_ix = [
            self.tokenizer.vocab.get(token, -1) for token in forbidden_tokens if token and token in self.tokenizer.vocab
        ]

        try:
            sep_tokens = self.tokenizer.encode(self.step_separator, add_special_tokens=False)
            if sep_tokens:
                sep_token_id = sep_tokens[0]
                token_to_id = {token: idx for token, idx in self.tokenizer.vocab.items()}
                id_to_token = {idx: token for token, idx in token_to_id.items()}
                sep_token_str = id_to_token.get(sep_token_id, "")
                self.tokens_containing_sep = {idx for token, idx in token_to_id.items() if sep_token_str in token}
            else:
                self.tokens_containing_sep = set()
        except Exception:
            self.tokens_containing_sep = set()

    # ===== CORE INTERFACE METHODS =====

    def get_step_prefix(self, agent_name: str, step_index: int, intervention_type: InterventionType | None = None) -> str:
        """Generate step prefix"""
        agent = next((a for a in self.agents if a.name == agent_name), None)

        if agent and agent.role == AgentRole.COORDINATOR:
            if intervention_type:
                return f"**{agent_name} [{intervention_type.value}-{step_index}]:** "
            else:
                return f"**{agent_name} [COORD-{step_index}]:** "
        else:
            return f"**{agent_name} [{step_index}]:** "

    def format_agent_list(self, agents: list[AgentConfig] = None) -> str:
        """Format agent list in natural language"""
        if agents is None:
            agents = self.agents

        names = [agent.name for agent in agents]
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return f"{', '.join(names[:-1])}, and {names[-1]}"

    def format_worker_list(self) -> str:
        """Format worker list"""
        return self.format_agent_list(self.workers) if self.workers else ""

    def get_full_prompt(self, problem: str, **kwargs) -> str:
        """Generate complete prompt for problem"""
        if self.pass_system_prompt_as_user_message:
            conversation = [{"role": "user", "content": self.system_prompt + self.step_separator + problem}]
        else:
            conversation = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": problem}]

        return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **kwargs)

    def is_end_of_step(self, worker_tokens: Sequence[int]) -> bool:
        """Check if worker finished step"""
        if not worker_tokens or worker_tokens[-1] not in self.tokens_containing_sep:
            return False

        try:
            step_string = self.tokenizer.decode(worker_tokens)
            step_string = step_string[: step_string.rindex(self.step_separator)].strip()
            return any(step_string.endswith(char) for char in self.end_of_step_chars)
        except Exception:
            return False

    def should_finish_reasoning(self, worker_tokens: Sequence[int]) -> bool:
        """Check if reasoning should stop"""
        try:
            decoded = self.tokenizer.decode(worker_tokens)
            return self.get_final_answer(decoded) is not None
        except Exception:
            return False

    def get_final_answer(self, response: str) -> T | None:
        """Extract final answer"""
        return find_last_valid_result(response, "\\boxed{", "}", self.extract_result)

    # ===== SYSTEM PROMPT GENERATION =====

    def _build_system_prompt(self) -> str:
        """Build complete system prompt"""
        sections = []

        # Header
        sections.append("# Multi-Agent Collaborative Reasoning")

        # Core rules
        sections.append(self._generate_core_rules())

        # Coordinator instructions (if applicable)
        if self.has_coordinator():
            sections.append(self._generate_coordinator_instructions())

        # Examples
        sections.append(self._generate_examples_section())

        # Collaboration strategies
        sections.append(self._generate_collaboration_strategies())

        # Final instructions
        sections.append(self._generate_final_instructions())

        return self.step_separator.join(sections)

    def _generate_core_rules(self) -> str:
        """Generate core collaboration rules"""
        num_agents = len(self.agents)
        agent_list = self.format_agent_list()

        base_rules = f"""You will collaborate on this problem with {num_agents - 1} other assistants. You will write your thoughts simultaneously with them and collaborate without redundant work. You can collaborate by doing different parts of the problem, double-checking each other's results, trying different approaches, or any other means.

There are {num_agents} assistants in total, including yourself. You will refer to each other as {agent_list}."""

        role_context = ""
        if self.has_coordinator():
            role_context = f"""

If you are the {self.coordinator.name}, you will coordinate the team effort, assign tasks, monitor progress, verify results, and determine when to provide the final answer. Working agents should focus on their assigned tasks and report progress to you."""

        workflow_rules = f"""

You will solve the problem together, writing your thoughts in parallel. You will be able to see each other's past and current thoughts as you write them. You will see each other's previous steps as {self.get_step_prefix("AssistantName", "step")}{self.pivot_message} .

In the '{self.history_header}' section, the automated system will gather the thoughts of all {num_agents} assistants as you write them.

After the '{self.work_in_progress_others}' section, you will see the other assistants' unfinished steps. They will write those steps concurrently with you. You will take into account what they are doing. If another assistant gives you suggestions, you should address them.

You will always see *other* assistants' incomplete thoughts first, and then, after '{self.work_in_progress_self}', your own current step. Other assistants will continue writing their thoughts in the background while you will continue writing your own.

Since you and others all write your thoughts in parallel, you will initially see only partial (unfinished) thoughts that others will continue in parallel, while you write yours. Others' thoughts will appear at the end of their unfinished step, near {self.pivot_message}. Other assistants may write new thoughts while you are writing yours.

You will use these partial thoughts to decide how best to collaborate without doing the same work twice. You will periodically check what other assistants are doing and you should adjust your actions based on what they are doing so you collaborate efficiently with them.

If what you are currently doing is the same thing that another assistant has already done or is in process of doing, you will stop and change to a different task right away, so as to avoid doing redundant work. With {num_agents} agents working together, there are many opportunities for parallel computation and verification."""

        final_answer_rule = ""
        if self.has_coordinator():
            final_answer_rule = f"""

Only the {self.coordinator.name} should provide the final answer using {self.final_answer_example} after verifying that the team has solved the problem correctly."""
        else:
            final_answer_rule = f"""

When you are done with the problem, any one of you ({agent_list}) can return the **final** answer as {self.final_answer_example}, after which, you will no longer be able to update it."""

        return base_rules + role_context + workflow_rules + final_answer_rule

    def _generate_coordinator_instructions(self) -> str:
        """Generate coordinator-specific instructions"""
        if not self.has_coordinator():
            return ""

        coord_name = self.coordinator.name
        worker_list = self.format_worker_list()

        return f"""## Special Role: {coord_name} (Coordinator)

As the **Coordinator**, {coord_name} has special responsibilities:

1. **Strategic Planning**: Analyze the problem and divide it into subtasks for {worker_list}
2. **Task Assignment**: Clearly assign specific parts of the problem to different working agents
3. **Real-time Monitoring**: Actively watch all agents' work and intervene when necessary
4. **Error Detection & Correction**: Immediately point out mathematical errors, logical flaws, or incorrect approaches
5. **Redundancy Prevention**: Stop agents from duplicating work by redirecting them to different tasks
6. **Progress Coordination**: Ensure agents build on each other's work effectively
7. **Verification**: Cross-check results from different agents and identify any inconsistencies
8. **Early Termination**: Detect when enough progress has been made to provide a confident final answer
9. **Final Answer Authority**: Only {coord_name} should provide the final {self.final_answer_example} after verifying all work

**Intervention Protocols:**
- **Error Intervention**: "{coord_name} [ERR-X]: Stop! [AgentName], there's an error in step X. The correct approach is..."
- **Redundancy Intervention**: "{coord_name} [INT-X]: [AgentName], [OtherAgent] is already working on that. Please switch to..."
- **Redirection**: "{coord_name} [REDIR-X]: [AgentName], since [OtherAgent] completed that part, please now work on..."
- **Clarification**: "{coord_name} [BUILD-X]: Good! Now use that result to..."

**Working Agent Response Requirements:**
When {coord_name} provides feedback, working agents ({worker_list}) must:
- **Acknowledge**: "Understood, {coord_name}. Switching to..."
- **Implement**: Immediately change their approach based on coordination guidance
- **Report**: Confirm completion of redirected tasks
- **Ask for help**: Request clarification if coordinator's guidance is unclear

Working agents should focus on their assigned tasks and actively respond to coordinator interventions."""

    def _generate_examples_section(self) -> str:
        """Generate examples section with dynamic content"""
        examples = self._create_examples()

        return f"""# Examples

## 1. Basic example of collaborating within one step

{examples["basic_coordination"]}

## 2. Full example with multiple agents

{examples["complex_problem"]}

# How to collaborate effectively with multiple agents

{self._generate_collaboration_strategies_text()}

**Strategizing with multiple agents:**

{examples["parallel_calculation"]}

**Multi-agent communication:**

{examples["method_comparison"]}

**Avoiding redundancy with multiple agents:**

{examples["redundancy_prevention_1"]}

{examples["redundancy_prevention_2"]}

**Real-time error correction:**

{examples["error_correction"]}

**Coordinated result building:**

{examples["result_building"]}"""

    def _generate_collaboration_strategies_text(self) -> str:
        """Generate collaboration strategies text"""
        num_agents = len(self.agents)

        if self.has_coordinator():
            return f"""You will take into account what the other assistants are doing and change your actions accordingly. Here is how you can collaborate effectively with {num_agents} agents:

**If you are the Coordinator ({self.coordinator.name}):**
- **Strategic planning:** Analyze the problem and break it into logical subtasks that can be distributed among working agents
- **Task assignment:** Clearly delegate specific parts to different agents based on their strengths or the problem structure  
- **Real-time intervention:** Actively monitor and intervene when you notice errors, redundancy, or inefficient approaches
- **Error correction:** Immediately point out mistakes with specific guidance
- **Redundancy prevention:** Interrupt duplicate work and redirect agents to new tasks
- **Progress redirection:** Guide agents when priorities change
- **Verification authority:** Cross-check all results and identify inconsistencies before providing the final answer
- **Early termination detection:** Recognize when sufficient progress has been made to confidently provide the final answer
- **Final answer responsibility:** Only provide the final answer after verifying the team's work is correct and complete

**If you are a Working Agent:**
- **Immediate response to coordination:** When the coordinator intervenes, immediately acknowledge and adjust
- **Error acknowledgment:** When corrected, thank the coordinator and implement the fix
- **Redundancy response:** When redirected from duplicate work, switch tasks immediately
- **Progress reporting:** Regularly update the coordinator on findings and ask for guidance when stuck
- **Collaborative building:** Build on other agents' verified work rather than starting from scratch
- **Deference to coordination:** Accept task reassignments and guidance from the coordinator gracefully

- **2. Active communication:** You constantly communicate about your progress, errors, and discoveries to avoid duplication and build on each other's work
- **3. Error checking:** You actively look for and point out errors in each other's work, with immediate acknowledgment and correction
- **4. Parallel processing:** You can split complex problems into {num_agents} or more subtasks and work on them simultaneously
- **5. Multi-method approaches:** With more agents, you can try multiple solution methods simultaneously and compare results for validation
- **6. Dynamic reallocation:** You quickly switch tasks when redundancy is detected or when priorities change
- **7. Verification chains:** You create systematic verification where agents double-check each other's work
- **8. Progressive refinement:** You build on each other's work progressively, with each agent adding sophistication or handling edge cases

**Communication Examples:**
- "Stop! Alice, your derivative calculation has an error. The chain rule gives us..."
- "Wait, Bob is already solving that equation. Charlie, please work on the boundary conditions instead."
- "Good progress everyone. Since we have X and Y confirmed, let's focus on verifying Z before concluding."
- "Alice, excellent work on part 1. Bob, please build on Alice's result for part 2."

To decide how best to collaborate, you will periodically assess the overall team progress and intervene when necessary to maintain efficiency and accuracy."""
        else:
            return f"""You will take into account what the other assistants are doing and change your actions accordingly. Here is how you can collaborate effectively with {num_agents} agents:

- **1. Strategic coordination:** With {num_agents} agents, you should think strategically about how to divide work. You can designate roles (e.g., one agent focuses on algebraic manipulation, another on verification, another on alternative approaches). If there's disagreement about strategy, you should all default to {self.agents[0].name}'s coordination suggestions.
- **2. Parallel processing:** You can split complex problems into {num_agents} or more subtasks and work on them simultaneously. This is especially effective for problems with multiple independent components.
- **3. Multi-method approaches:** With more agents, you can try multiple solution methods simultaneously (analytical, numerical, geometric approaches) and compare results for validation.
- **4. Communication and coordination:** You can ask each other questions, give suggestions, and coordinate your efforts (e.g. '{self.agents[0].name}, should I verify your step 3 or work on the next equation?').
- **5. Work announcement:** You can announce what you will do next to avoid conflicts (e.g. 'I will handle the integration while others work on the algebraic part').
- **6. Dynamic reallocation:** If you notice redundancy or that your current task is no longer optimal, you should stop and switch to a more valuable contribution.
- **7. Verification chains:** With multiple agents, you can create verification chains where different agents double-check each other's work systematically.
- **8. Progressive refinement:** You can build on each other's work progressively, with each agent adding layers of sophistication or handling edge cases.

With {num_agents} agents, coordination becomes more important but also enables more sophisticated parallel problem-solving strategies.

To decide how best to collaborate, you will periodically assess not just what you are doing, but how your work fits into the overall team effort and whether there are higher-value tasks you should be pursuing instead."""

    def _generate_collaboration_strategies(self) -> str:
        """Generate collaboration strategies section"""
        return ""  # Already included in examples section

    def _generate_final_instructions(self) -> str:
        """Generate final problem-solving instructions"""
        agent_list = self.format_agent_list()

        if self.has_coordinator():
            return f"""# Solve the following problem

{agent_list}, you will now solve the next problem together. With {self.coordinator.name} coordinating the effort, keep track of who does what work and communicate effectively. The coordinator will monitor progress and determine when to provide the final answer."""
        else:
            return f"""# Solve the following problem

{agent_list}, you will now solve the next problem together. Keep track of who does what work and communicate to avoid doing the same work twice. With {len(self.agents)} agents working together, coordination and clear communication are essential for efficiency."""

    # ===== DYNAMIC EXAMPLES GENERATION =====

    def _create_examples(self) -> dict[str, str]:
        """Create all examples based on current configuration"""

        def make_example(question: str, answer: str) -> str:
            return f"<example>\n\n{question}\n\n{answer}\n\n</example>"

        examples = {}

        # Basic coordination example
        if self.has_coordinator():
            coord = self.coordinator.name
            workers = [w.name for w in self.workers[:2]]  # Use first 2 workers

            examples["basic_coordination"] = make_example(
                "Solve three problems: 1) Ann has 2 apples, Mark has 5 apples. How many total? 2) Solve x + y = 4 if y = 5. 3) Calculate 15% of 240.",
                f"""{self.begin_of_reasoning}{self.step_separator}{self.history_header}{self.step_separator}{self.work_in_progress_others}

{self.get_step_prefix(coord, 1)}I'll coordinate our approach to these three problems. {workers[0]}, please handle problem 1 (apple counting). {workers[1] if len(workers) > 1 else workers[0]}, take problem 2 (equation solving). I'll monitor progress and work on problem 3 (percentage calculation) while verifying your results.

{self.get_step_prefix(workers[0], 1)}Understood! Problem 1: Ann has 2 apples, Mark has 5 apples. Total = 2 + 5 = 7 apples. {coord}, this is complete.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 1)}Working on problem 2: x + y = 4, if y = 5. Substituting: x + 5 = 4, so x = 4 - 5 = -1. {coord}, please verify this result.{self.pivot_message}

{self.work_in_progress_self}

{self.get_step_prefix(coord, 2)}Excellent work! Let me verify: Problem 1 ✓ (2+5=7), Problem 2 ✓ (x=-1 when y=5). Problem 3: 15% of 240 = 0.15 × 240 = 36. 

All problems solved correctly. Final answers: 1) 7 apples, 2) x = -1, 3) 36.""",
            )
        else:
            workers = [a.name for a in self.agents[:3]]  # Use first 3 agents

            examples["basic_coordination"] = make_example(
                "Solve three problems: 1) Ann has 2 apples, Mark has 5 apples. How many total? 2) Solve x + y = 4 if y = 5. 3) Calculate 15% of 240.",
                f"""{self.begin_of_reasoning}{self.step_separator}{self.history_header}{self.step_separator}{self.work_in_progress_others}

{self.get_step_prefix(workers[0], 1)}I'll coordinate our work. {workers[1] if len(workers) > 1 else "everyone"}, please handle problem 1 (apples). {workers[2] if len(workers) > 2 else workers[1] if len(workers) > 1 else "I"}, take problem 2 (equation). I'll work on problem 3 (percentage calculation).

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 1)}Got it! Problem 1: Ann has 2 apples, Mark has 5 apples. Total = 2 + 5 = 7 apples.

{self.get_step_prefix(workers[2] if len(workers) > 2 else workers[1] if len(workers) > 1 else workers[0], 1)}Working on problem 2: x + y = 4, if y = 5. Substituting: x + 5 = 4, so x = 4 - 5 = -1.{self.pivot_message}

{self.work_in_progress_self}

{self.get_step_prefix(workers[0], 2)}Problem 3: 15% of 240 = 0.15 × 240 = 36. Great teamwork! Final answers: 1) 7 apples, 2) x = -1, 3) 36.""",
            )

        # Complex problem example
        if self.has_coordinator():
            coord = self.coordinator.name
            workers = [w.name for w in self.workers[:3]]

            examples["complex_problem"] = make_example(
                "Solve the system: x² + y² = 25, x + y = 7, and find all real solutions.",
                f"""{self.begin_of_reasoning}{self.step_separator}{self.history_header}

{self.get_step_prefix(coord, 1)}I'll coordinate our approach to this system of equations. {workers[0]}, use substitution method. {workers[1] if len(workers) > 1 else workers[0]}, try elimination approach. {workers[2] if len(workers) > 2 else "I"} will work on algebraic verification. I'll monitor progress and determine when we have a complete solution.

{self.get_step_prefix(workers[0], 1)}Using substitution: From x + y = 7, I get y = 7 - x. Substituting into x² + y² = 25: x² + (7-x)² = 25.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 1)}Using elimination: From x + y = 7, squaring both sides: (x + y)² = 49, so x² + 2xy + y² = 49. Since x² + y² = 25, we have 49 - 25 = 2xy, so xy = 12.

{self.get_step_prefix(workers[0], 2)}Expanding my equation: x² + 49 - 14x + x² = 25, so 2x² - 14x + 24 = 0, which gives x² - 7x + 12 = 0. Factoring: (x-3)(x-4) = 0, so x = 3 or x = 4.

{self.work_in_progress_others}

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 2)}My approach gives the same constraint: x + y = 7 and xy = 12. This leads to the quadratic t² - 7t + 12 = 0 with solutions t = 3, 4.{self.pivot_message}

{self.work_in_progress_self}

{self.get_step_prefix(coord, 2)}Excellent! Both approaches converge. The solutions are (3,4) and (4,3). Let me verify: 3² + 4² = 25 ✓, 3 + 4 = 7 ✓. \\boxed{{(3,4), (4,3)}}""",
            )
        else:
            workers = [a.name for a in self.agents[:3]]

            examples["complex_problem"] = make_example(
                "Solve the system: x² + y² = 25, x + y = 7, and find all real solutions.",
                f"""{self.begin_of_reasoning}{self.step_separator}{self.history_header}

{self.get_step_prefix(workers[0], 1)}Let's coordinate our approach. {workers[1] if len(workers) > 1 else "I"}, please use substitution method. {workers[2] if len(workers) > 2 else "I"} will try elimination and verify. I'll organize results.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 1)}Using substitution: From x + y = 7, we get y = 7 - x. Substituting into x² + y² = 25: x² + (7-x)² = 25.

{self.get_step_prefix(workers[0], 2)}Using elimination: (x + y)² = 49 gives xy = 12. Combined with x + y = 7, solutions are (3,4) and (4,3). \\boxed{{(3,4), (4,3)}}""",
            )

        # Parallel calculation example
        examples["parallel_calculation"] = make_example(
            "Calculate S(x) = x + x² + x³ + x⁴ + x⁵ for x = 1, 2, 3, ..., 12.",
            f"""{self.begin_of_reasoning}{self.step_separator}{self.history_header}

{self.get_step_prefix(self.agents[0].name, 1)}Let's divide this efficiently among {len(self.agents)} agents. I'll take x=1-3. {self.agents[1].name if len(self.agents) > 1 else "We"} can take x=4-6. We can work in parallel.

{self.get_step_prefix(self.agents[1].name if len(self.agents) > 1 else self.agents[0].name, 1)}Sounds good! I'll start with x=4: S(4) = 4 + 16 + 64 + 256 + 1024 = 1364.

{self.work_in_progress_others}

{self.get_step_prefix(self.agents[0].name, 2)}For x=1: S(1) = 5. For x=2: S(2) = 62. Excellent parallel work!{self.pivot_message}

{self.work_in_progress_self}

{self.get_step_prefix(self.agents[0].name, 3)}All agents are efficiently computing their assigned values.""",
        )

        # Method comparison example
        examples["method_comparison"] = make_example(
            "Find the area of triangle with vertices A(1,2), B(4,6), C(7,2).",
            f"""{self.begin_of_reasoning}{self.step_separator}{self.history_header}

{self.get_step_prefix(self.agents[0].name, 1)}Let's try multiple approaches. {self.agents[1].name if len(self.agents) > 1 else "I"} will use the coordinate formula. I'll try base-height method.

{self.get_step_prefix(self.agents[1].name if len(self.agents) > 1 else self.agents[0].name, 1)}Using coordinate formula: Area = ½|1(6-2) + 4(2-2) + 7(2-6)| = ½|4 + 0 - 28| = ½|-24| = 12.

{self.get_step_prefix(self.agents[0].name, 2)}Base-height method: Base AC = 6, Height = 4. Area = ½ × 6 × 4 = 12. Both methods agree! \\boxed{{12}}""",
        )

        # Redundancy prevention examples
        if self.has_coordinator():
            coord = self.coordinator.name
            workers = [w.name for w in self.workers[:2]]

            examples["redundancy_prevention_1"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.get_step_prefix(workers[0], 5)}I'm computing the derivative of f(x) = x³ + 2x² - 5x + 1. f'(x) = 3x² + 4x - 5.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 3)}Let me also find the derivative: f'(x) = 3x² + 4x - 5. 

{self.get_step_prefix(coord, 1, InterventionType.INTERRUPT)}Stop! {workers[1] if len(workers) > 1 else workers[0]}, {workers[0]} just computed that exact derivative. You're duplicating work. Please switch to finding the critical points using {workers[0]}'s result instead.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 4)}Understood, {coord}! Switching to critical points. Setting f'(x) = 0: 3x² + 4x - 5 = 0.

{self.work_in_progress_others}

{self.get_step_prefix(coord, 2)}Good coordination! {workers[0]} has the derivative, {workers[1] if len(workers) > 1 else "we"} are finding critical points. Excellent division of work.{self.pivot_message}

{self.work_in_progress_self}

{self.get_step_prefix(workers[0], 6)}Perfect! Now I'll help solve 3x² + 4x - 5 = 0 using the quadratic formula.{self.step_separator}</example>"""

            examples["redundancy_prevention_2"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.work_in_progress_others}

{self.get_step_prefix(workers[0], 4)}Computing integral ∫(2x + 3)dx = x² + 3x + C. 

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 3)}I'll compute ∫(2x + 3)dx as well...

{self.get_step_prefix(coord, 1, InterventionType.INTERRUPT)}Wait! {workers[1] if len(workers) > 1 else workers[0]}, {workers[0]} is already handling that integral. Please compute ∫(x² - 1)dx instead to avoid duplication.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 4)}You're right, {coord}! Switching to ∫(x² - 1)dx = x³/3 - x + C.

{self.work_in_progress_self}

{self.get_step_prefix(coord, 2)}Perfect division of work! I'll coordinate the final results once both integrals are complete.{self.step_separator}</example>"""
        else:
            workers = [a.name for a in self.agents[:3]]

            examples["redundancy_prevention_1"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.get_step_prefix(workers[0], 5)}I'm computing the derivative of f(x) = x³ + 2x² - 5x + 1. f'(x) = 3x² + 4x - 5.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 3)}Let me also find the derivative: f'(x) = 3x² + 4x - 5. Wait, {workers[0]} just computed the same thing! I should work on finding critical points instead.

{self.work_in_progress_others}

{self.get_step_prefix(workers[0], 6)}Good coordination! Now I'll find where f'(x) = 0: 3x² + 4x - 5 = 0.{self.step_separator}</example>"""

            examples["redundancy_prevention_2"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.work_in_progress_others}

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 4)}Computing integral ∫(2x + 3)dx = x² + 3x + C. 

{self.get_step_prefix(workers[2] if len(workers) > 2 else workers[1] if len(workers) > 1 else workers[0], 3)}I see {workers[1] if len(workers) > 1 else "someone"} is handling that integral. I'll compute ∫(x² - 1)dx = x³/3 - x + C instead.

{self.work_in_progress_self}

{self.get_step_prefix(workers[0], 5)}Perfect division of work! I'll coordinate the final results once everyone finishes their integrals.{self.step_separator}</example>"""

        # Error correction example
        if self.has_coordinator():
            coord = self.coordinator.name
            workers = [w.name for w in self.workers[:2]]

            examples["error_correction"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.get_step_prefix(workers[0], 3)}Computing the definite integral: ∫₀² x² dx = [x³/3]₀² = 8/3 - 0 = 8/3.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 2)}Building on that result, the area under the curve from 0 to 2 is 8/3. Now I'll compute the volume of revolution...

{self.get_step_prefix(coord, 1, InterventionType.ERROR)}Wait! Let me double-check that integral: ∫₀² x² dx = [x³/3]₀² = 2³/3 - 0³/3 = 8/3. Actually, {workers[0]} is correct!

{self.get_step_prefix(workers[0], 4)}Thanks for verifying, {coord}! The calculation is indeed 8/3.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 3)}Continuing with the volume calculation using the confirmed result of 8/3...

{self.work_in_progress_others}

{self.get_step_prefix(coord, 2)}Good! Both agents are building correctly on the verified integral result. {workers[1] if len(workers) > 1 else "Continue"} with the volume calculation.{self.pivot_message}

{self.work_in_progress_self}

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 4)}Volume = π∫₀² (x²)² dx = π∫₀² x⁴ dx = π[x⁵/5]₀² = π(32/5) = 32π/5.{self.step_separator}</example>"""
        else:
            workers = [a.name for a in self.agents[:2]]

            examples["error_correction"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.get_step_prefix(workers[0], 3)}Computing the definite integral: ∫₀² x² dx = [x³/3]₀² = 8/3 - 0 = 8/3.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 2)}Let me double-check: ∫₀² x² dx = [x³/3]₀² = 2³/3 = 8/3. That's correct!

{self.get_step_prefix(workers[0], 4)}Thanks for the verification, {workers[1] if len(workers) > 1 else "everyone"}! Now we can confidently use 8/3 for the next steps.{self.step_separator}</example>"""

        # Result building example
        if self.has_coordinator():
            coord = self.coordinator.name
            workers = [w.name for w in self.workers[:2]]

            examples["result_building"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.get_step_prefix(workers[0], 2)}I found that the first equation gives us x = 3 or x = -1.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 1)}From the second equation, I'm getting y = 2x + 5.

{self.get_step_prefix(coord, 1, InterventionType.BUILD)}Excellent! {workers[1] if len(workers) > 1 else workers[0]}, now substitute {workers[0]}'s solutions into your equation. If x = 3, then y = 2(3) + 5 = 11. If x = -1, then y = 2(-1) + 5 = 3.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 2)}Perfect! So our solutions are (3, 11) and (-1, 3). Let me verify these in the original equations.

{self.get_step_prefix(coord, 2)}Good teamwork! {workers[0]} found the x-values, {workers[1] if len(workers) > 1 else "we"} found the y-relationship, and now we're verifying. This is efficient collaboration.

{self.work_in_progress_others}

{self.get_step_prefix(workers[0], 3)}I'll help verify: For (3,11) in first equation... ✓. For (-1,3) in first equation... ✓{self.pivot_message}

{self.work_in_progress_self}

{self.get_step_prefix(coord, 3, InterventionType.FINAL)}Both solutions verified! Final answer: (3,11) and (-1,3). \\boxed{{(3,11), (-1,3)}}{self.step_separator}</example>"""
        else:
            workers = [a.name for a in self.agents[:2]]

            examples["result_building"] = f"""<example>{self.step_separator}(previous steps omitted)

{self.get_step_prefix(workers[0], 2)}I found that the first equation gives us x = 3 or x = -1.

{self.get_step_prefix(workers[1] if len(workers) > 1 else workers[0], 1)}From the second equation, I'm getting y = 2x + 5. Building on {workers[0]}'s result: if x = 3, then y = 11. If x = -1, then y = 3.

{self.get_step_prefix(workers[0], 3)}Great building! Let me verify: (3,11) and (-1,3) both check out in the original equations. \\boxed{{(3,11), (-1,3)}}{self.step_separator}</example>"""

        return examples


def find_last_valid_result(response: str, prefix: str, suffix: str, extract_result: Callable[[str], T]) -> T | None:
    """Find the rightmost valid result between prefix and suffix"""
    remaining_text = response

    while True:
        try:
            start = remaining_text.rindex(prefix)
            try:
                end = remaining_text.index(suffix, start)
                candidate = remaining_text[start + len(prefix) : end]
                return extract_result(candidate)
            except (ValueError, Exception):
                remaining_text = remaining_text[:start]
        except ValueError:
            return None


# ===== CONVENIENCE FUNCTIONS =====


def create_coordinator_team(
    coordinator_name: str = "Coordinator", worker_names: list[str] = None, tokenizer: transformers.PreTrainedTokenizer = None, **kwargs
) -> MultiAgentFormatter:
    """Create a team with one coordinator and multiple workers"""
    if worker_names is None:
        worker_names = ["Alice", "Bob", "Charlie"]

    agents = [AgentConfig(coordinator_name, AgentRole.COORDINATOR)]
    agents.extend([AgentConfig(name, AgentRole.WORKER) for name in worker_names])

    return MultiAgentFormatter(tokenizer, agents, **kwargs)


def create_peer_team(agent_names: list[str] = None, tokenizer: transformers.PreTrainedTokenizer = None, **kwargs) -> MultiAgentFormatter:
    """Create a team of peer workers without a coordinator"""
    if agent_names is None:
        agent_names = ["Alice", "Bob", "Charlie"]

    agents = [AgentConfig(name, AgentRole.WORKER) for name in agent_names]

    return MultiAgentFormatter(tokenizer, agents, **kwargs)


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Mock tokenizer for demonstration
    class MockTokenizer:
        def __init__(self):
            self.vocab = {"#": 1, "<think>": 2, "</think>": 3, "\n\n": 4}
            self.bos_token = "<s>"
            self.eos_token = "</s>"

        def encode(self, text, add_special_tokens=False):
            return [4] if text == "\n\n" else [1, 2, 3]

        def decode(self, tokens):
            return "mock decoded text"

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, **kwargs):
            content = conversation[0]["content"]
            return f"<s>User: {content}</s>" if tokenize == False else [1, 2, 3]

    # Example 1: Coordinator team
    print("=== Creating Coordinator Team ===")
    tokenizer = MockTokenizer()

    team = create_coordinator_team(coordinator_name="Manager", worker_names=["Alice", "Bob", "Charlie"], tokenizer=tokenizer)

    print(f"Team has coordinator: {team.has_coordinator()}")
    print(f"Coordinator: {team.coordinator.name if team.coordinator else 'None'}")
    print(f"Workers: {[w.name for w in team.workers]}")
    print(f"All agents: {team.format_agent_list()}")

    # Example 2: Generate a complete prompt
    print("\n=== Generating Complete Prompt ===")
    problem = "Solve the quadratic equation x² - 5x + 6 = 0"
    prompt = team.get_full_prompt(problem)

    print(f"Problem: {problem}")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Sample of prompt:\n{prompt[:500]}...")

    # Example 3: Test step prefixes
    print("\n=== Testing Step Prefixes ===")
    coord_prefix = team.get_step_prefix("Manager", 1)
    worker_prefix = team.get_step_prefix("Alice", 2)
    error_prefix = team.get_step_prefix("Manager", 3, InterventionType.ERROR)
    interrupt_prefix = team.get_step_prefix("Manager", 4, InterventionType.INTERRUPT)

    print(f"Coordinator step: {coord_prefix}")
    print(f"Worker step: {worker_prefix}")
    print(f"Error intervention: {error_prefix}")
    print(f"Interrupt intervention: {interrupt_prefix}")

    # Example 4: Peer team
    print("\n=== Creating Peer Team ===")
    peer_team = create_peer_team(agent_names=["Alice", "Bob"], tokenizer=tokenizer)

    print(f"Peer team has coordinator: {peer_team.has_coordinator()}")
    print(f"Peer team agents: {peer_team.format_agent_list()}")

    # Example 5: Answer extraction
    print("\n=== Testing Answer Extraction ===")
    test_response = "After solving, we get x = 2 and x = 3. So the answer is \\boxed{2, 3} which is our final result."
    answer = team.get_final_answer(test_response)
    print(f"Test response: {test_response}")
    print(f"Extracted answer: {answer}")

    print("\n=== All Tests Completed Successfully! ===")
