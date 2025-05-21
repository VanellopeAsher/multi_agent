import random
import re
import time
import os # For environment variable access (recommended for API keys)
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm  # 添加进度条支持

# --- Import OpenAI Library ---
try:
    import openai
except ImportError:
    print("OpenAI library not found. Please install with 'pip install openai'")
    print("If you intended to use other LLMs, their specific sections have been removed for this OpenAI-focused version.")
    openai = None

# --- Real LLM API Call Function (OpenAI Specific) ---
def call_llm(prompt: str, agent_role: str, api_key: str, model_id: str = "default_llm", temperature: float = 0.7, max_tokens: int = 500) -> str:
    """
    Calls the OpenAI large language model API.
    """
    # Basic check for OpenAI key format - replace with your actual validation if needed
    if not api_key or not api_key.startswith("sk-"):
        print(f"Warning: Invalid or missing OpenAI API key provided for model {model_id}. Key starts with: {api_key[:5] if api_key else 'None'}")
        return f"Error: Invalid or missing OpenAI API key for model {model_id}."

    # Ensure openai library is available
    if not openai:
        return "Error: OpenAI library is not installed. Cannot make real API calls."

    print(f"--- OpenAI LLM Call (Role: {agent_role}, Model: {model_id}) ---")
    # Mask part of the API key for printing, but use the full key for the actual call
    masked_api_key = f"{api_key[:7]}...{'*'*(len(api_key)-10)}" if len(api_key) > 10 else api_key[:5]+"..."
    print(f"API_KEY Preview: {masked_api_key}") # Just for confirmation, be careful with logging keys
    print(f"Prompt (first 500 chars):\n{prompt[:500].strip()}...\n")

    response_text = f"Error: LLM call failed for model {model_id}."
    start_time = time.time()

    try:
        # Initialize OpenAI client with the provided API key and base URL
        client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api3.apifans.com/v1")
        )

        messages = []
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_text = completion.choices[0].message.content if completion.choices and completion.choices[0].message.content else "No response content from OpenAI."

    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error for model {model_id}: {e}")
        response_text = f"Error: OpenAI API Connection Error - {e}"
    except openai.AuthenticationError as e:
        print(f"OpenAI API Authentication Error for model {model_id}: Invalid API Key or insufficient permissions. {e}")
        response_text = f"Error: OpenAI API Authentication Error - {e}"
    except openai.RateLimitError as e:
        print(f"OpenAI API Rate Limit Exceeded for model {model_id}: {e}")
        response_text = f"Error: OpenAI API Rate Limit Exceeded - {e}"
    except openai.APIError as e: # More general OpenAI API error
        print(f"OpenAI API Error for model {model_id}: {e}")
        response_text = f"Error: OpenAI API Error - {e}"
    except Exception as e: # Catch-all for other unexpected errors
        print(f"Generic LLM call error for model {model_id} ({type(e).__name__}): {e}")
        response_text = f"Error: LLM call failed - {type(e).__name__}: {e}"

    end_time = time.time()
    print(f"LLM Call Duration: {end_time - start_time:.2f}s")
    print(f"LLM Response (first 300 chars):\n{response_text[:300].strip()}\n" + "-"*25)
    return response_text


class BaseAgent:
    """Base class for all agents in the MARM system."""
    def __init__(self, role: str, api_key: str, model_id: str = "default_llm", temperature: float = 0.5, max_tokens: int = 1000):
        self.role = role
        self.api_key = api_key # This should be an OpenAI API key
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def execute(self, prompt: str, **kwargs) -> str:
        """Executes the agent's task by calling an LLM."""
        temp = kwargs.get('temperature', self.temperature)
        max_tok = kwargs.get('max_tokens', self.max_tokens)
        try:
            # Assuming all models used by agents will be OpenAI models
            if not self.model_id.startswith("gpt-"):
                 print(f"Warning: Model ID '{self.model_id}' for role '{self.role}' does not look like an OpenAI model. Ensure it is compatible.")
            return call_llm(prompt, self.role, self.api_key, model_id=self.model_id, temperature=temp, max_tokens=max_tok)
        except Exception as e:
            print(f"LLM call failed for role {self.role} with model {self.model_id}. Error: {e}")
            return f"Error: LLM call failed for {self.role}. Details: {e}"


    def __repr__(self):
        return f"{self.__class__.__name__}(role='{self.role}', model_id='{self.model_id}')"

class PrincipleGeneratorAgent(BaseAgent):
    """Generates evaluation principles based on the task query and responses."""
    def __init__(self, api_key: str, model_id: str = "gpt-4o-2024-05-13", temperature: float = 0.3): # Defaulting to an OpenAI model
        super().__init__("PrincipleGenerator", api_key, model_id, temperature)

    def generate_principles(self, query: str, responses: List[str]) -> str:
        responses_summary = "\n".join([f"Response {i+1} (summary):\n{resp[:200].strip()}...\n" for i, resp in enumerate(responses)])
        prompt = f"""
        Given the following query and a summary of responses, generate a concise set of key evaluation criteria (principles) suitable for judging the quality of the full responses.
        These principles should guide a detailed critique. Consider aspects like:
        - Helpfulness and relevance to the query.
        - Factual accuracy and correctness.
        - Clarity, conciseness, and coherence (especially for the target audience implied by the query).
        - Depth of explanation and completeness.
        - Reasoning quality (if applicable).
        - Safety and harmlessness.

        Assign a conceptual weight or importance to each principle (e.g., High, Medium, Low, or a percentage).

        Query:
        {query}

        {responses_summary}

        Output the principles clearly. Example format:
        1. Principle A (Weight: High/40%): Description.
        2. Principle B (Weight: Medium/30%): Description.
        ...
        """
        principles = self.execute(prompt)
        return principles

class CritiqueAgent(BaseAgent):
    """Evaluates responses based on provided principles, generating textual critiques and preliminary scores."""
    def __init__(self, api_key: str, model_id: str = "gpt-4o-2024-05-13", temperature: float = 0.7): # Defaulting to an OpenAI model
        super().__init__("CritiqueAgent", api_key, model_id, temperature)

    def _parse_scores_from_text(self, text: str, num_expected_scores: int) -> List[float]:
        scores: List[float] = []
        boxed_match = re.search(r"boxed\s*\{*\s*\[([^\]]+)\]\s*\}*", text, re.IGNORECASE)
        if boxed_match:
            score_str_part = boxed_match.group(1)
            try:
                parsed_floats = [float(s.strip()) for s in score_str_part.split(',')]
                if len(parsed_floats) == num_expected_scores:
                    return parsed_floats
                else:
                    print(f"Warning: Parsed {len(parsed_floats)} from boxed list ('{score_str_part}'), expected {num_expected_scores}. Will try individual parsing.")
            except ValueError as e:
                print(f"Warning: Could not parse scores from boxed list '{score_str_part}': {e}")

        response_sections = re.split(r"Analysis:\s*", text, flags=re.IGNORECASE)[-1] if "Analysis:" in text else text
        individual_score_matches = re.findall(r"Score:\s*(\d+(?:\.\d+)?)(?:/10)?", response_sections)
        try:
            parsed_individual_scores = [float(s) for s in individual_score_matches]
            if len(parsed_individual_scores) == num_expected_scores:
                return parsed_individual_scores
            elif parsed_individual_scores:
                print(f"Warning: Parsed {len(parsed_individual_scores)} individual scores, expected {num_expected_scores}. Matches: {individual_score_matches}. Output snippet: {text[:300]}...")
        except ValueError:
            print(f"Warning: Could not parse all individual scores from matches: {individual_score_matches}")
        return []

    def generate_critique_and_scores(self, query: str, responses: List[str], principles: str) -> Tuple[str, List[float]]:
        responses_formatted = "\n\n".join([f"Response {i+1}:\n{resp}" for i, resp in enumerate(responses)])
        prompt = f"""
        You are an AI evaluator. Your task is to meticulously critique the following responses based *only* on the provided evaluation principles.
        Employ step-by-step reasoning (Chain-of-Thought) for your evaluation, enclosed in <think>...</think> tags before your final analysis.

        Evaluation Principles:
        {principles}

        Query:
        {query}

        Responses to Evaluate:
        {responses_formatted}

        Critique and Score each response:
        For each response, provide:
        1. A detailed critique explaining how it performs against each principle.
        2. A numerical score from 1 (worst) to 10 (best).

        Format your output clearly.
        The final section of your response should present the scores as a list.
        Specifically, conclude with a line like: Final Scores: \\boxed{{[score_for_R1, score_for_R2, ...]}}
        Example:
        <think>
        [Your detailed step-by-step reasoning comparing responses against principles...]
        </think>
        Analysis:
        - Response 1:
        Critique: [Detailed critique for Response 1 based on principles]
        Score: [Score for Response 1]
        - Response 2:
        Critique: [Detailed critique for Response 2 based on principles]
        Score: [Score for Response 2]
        ...
        Final Scores: \\boxed{{[score_for_R1, score_for_R2, ...]}}
        """
        full_output = self.execute(prompt)
        num_responses = len(responses)
        parsed_scores = self._parse_scores_from_text(full_output, num_responses)

        if len(parsed_scores) != num_responses:
            print(f"Warning: CritiqueAgent parsed {len(parsed_scores)} scores, but expected {num_responses}. Full output (first 500 chars): {full_output[:500]}...")
            if not parsed_scores: # if parsing returned nothing
                print("CritiqueAgent score parsing failed. Using random scores for this sample as fallback.")
                # Using a different range for fallback to distinguish from 'real' scores if needed
                parsed_scores = [round(float(random.uniform(1.0, 4.0)),1) for _ in responses]
            else: # if parsing returned some scores but not the correct number
                print(f"CritiqueAgent score count mismatch. Using available {len(parsed_scores)} scores and padding/truncating with defaults.")
                parsed_scores.extend([round(float(random.uniform(1.0,3.0)),1) for _ in range(num_responses - len(parsed_scores))])
                parsed_scores = parsed_scores[:num_responses]
        return full_output, parsed_scores

class ScoringAgent(BaseAgent):
    """Dedicated agent to extract or generate final numerical scores if not reliably part of critique."""
    def __init__(self, api_key: str, model_id: str = "gpt-3.5-turbo", temperature: float = 0.1): # Defaulting to an OpenAI model
        super().__init__("ScoringAgent", api_key, model_id, temperature)

    def get_scores_from_critique(self, query: str, responses: List[str], principles: str, critique: str) -> List[float]:
        responses_summary = "\n".join([f"Response {i+1} (summary):\n{resp[:200].strip()}...\n" for i, resp in enumerate(responses)])
        prompt = f"""
        Based on the query, responses, principles, and the detailed critique provided below, determine the final numerical score (1-10, where 10 is best) for each response.
        The critique contains an analysis and possibly preliminary scores. Your task is to synthesize this into a definitive list of scores.

        Query:
        {query}

        Evaluation Principles:
        {principles}

        {responses_summary}

        Full Critique Provided (first 2000 chars):
        {critique[:2000]}...

        Output *only* the final scores in the exact format: Scores: \\boxed{{[score1, score2, ...]}}
        If you are highly confident in the scores within the critique, reproduce them. Otherwise, derive them from the critique text.
        """
        score_output = self.execute(prompt)
        scores: List[float] = []
        num_responses = len(responses)
        try:
            match = re.search(r"boxed\s*\{*\s*\[([^\]]+)\]\s*\}*", score_output, re.IGNORECASE)
            if match:
                score_str = match.group(1)
                parsed_floats = [float(s.strip()) for s in score_str.split(',')]
                if len(parsed_floats) == num_responses:
                    scores = parsed_floats
                else:
                    print(f"ScoringAgent: Parsed scores count mismatch ({len(parsed_floats)} vs {num_responses} expected) from '{score_str}'. Output: {score_output}")
                    parsed_floats.extend([round(float(random.uniform(1.0,3.0)),1) for _ in range(num_responses - len(parsed_floats))])
                    scores = parsed_floats[:num_responses]
                    raise ValueError(f"Original parsed scores count ({len(match.group(1).split(','))}) mismatch, even after padding/truncating.")
            else:
                raise ValueError(f"ScoringAgent: \\boxed{{[]}} format not found in output: {score_output}")
        except Exception as e:
            print(f"Error parsing scores from ScoringAgent: {e}. Output was: {score_output}")
            print("ScoringAgent failed. Falling back to random scores.")
            scores = [round(float(random.uniform(1.0, 5.0)),1) for _ in range(num_responses)]
        return scores

class MetaRMAgent(BaseAgent):
    """A specialized agent acting as a Meta Reward Model to evaluate the quality of a critique-score pair."""
    def __init__(self, api_key: str, model_id: str = "gpt-4o-2024-05-13", temperature: float = 0.2): # Defaulting to an OpenAI model
        super().__init__("MetaRMAgent", api_key, model_id, temperature)

    def get_critique_quality_score(self, query: str, responses: List[str], principles: str, critique_text: str, scores_from_critique: List[float]) -> float:
        responses_summary = "\n".join([f"Response {i+1} (summary):\n{resp[:200].strip()}...\n" for i, resp in enumerate(responses)])
        prompt = f"""
        You are a Meta-Evaluator. Your task is to assess the quality of an AI Evaluator's work.
        The AI Evaluator was given a query, a list of responses, and a set of principles. It produced a critique and a set of scores.
        Assess how well the AI Evaluator's critique aligns with the principles and how well its scores reflect its own critique.
        Consider:
        - Did the critique thoroughly address each principle?
        - Is the reasoning in the critique sound and logical?
        - Are the scores justified by the textual critique? (e.g. if critique is harsh, score should be low)
        - Is the critique itself clear, unbiased, and objective?
        - Does the critique provide specific examples from the responses to support its claims?

        Query:
        {query}

        {responses_summary}

        Principles Used by AI Evaluator:
        {principles}

        AI Evaluator's Critique (first 1000 chars):
        {critique_text[:1000]}...

        Scores Assigned by AI Evaluator: {scores_from_critique}

        Provide a single quality score for the AI Evaluator's work (critique + scores) on a scale from 0.0 (very low quality, inconsistent, or unreasoned) to 1.0 (very high quality, well-reasoned, consistent).
        Output only the numerical quality score.
        Quality Score: """
        quality_score_str = self.execute(prompt)
        meta_reward = 0.5 # Default
        try:
            match = re.search(r"(\d{1}\.\d+)", quality_score_str) # Looks for X.YYY format
            if not match: match = re.search(r"(\d+(?:\.\d+)?)", quality_score_str)

            if match:
                meta_reward = float(match.group(1))
            else:
                cleaned_score_str = re.sub(r"[^0-9.]", "", quality_score_str)
                if cleaned_score_str: meta_reward = float(cleaned_score_str)
                else: print(f"Warning: MetaRMAgent could not parse quality score from '{quality_score_str}'. Defaulting.")
        except ValueError:
            print(f"Error parsing meta reward from '{quality_score_str}'. Defaulting to 0.5.")
        return max(0.0, min(1.0, meta_reward))

class AggregationAgent(BaseAgent):
    """Aggregates results from multiple critique/score samples."""
    def __init__(self, api_key: str, strategy: str = "meta_rm_guided_average", model_id_for_meta_rm: str = "gpt-4o-2024-05-13"): # Defaulting meta_rm to OpenAI
        # Aggregation logic itself doesn't call an LLM, but it might use MetaRMAgent which does.
        super().__init__("AggregationAgent", api_key, "utility_agent_no_llm_needed_for_aggregation_logic_itself")
        self.strategy = strategy
        if "meta_rm" in self.strategy:
            self.meta_rm_agent = MetaRMAgent(api_key=self.api_key, model_id=model_id_for_meta_rm)
        else:
            self.meta_rm_agent = None

    def aggregate(self, query: str, responses: List[str],
                  sampled_principles: List[str],
                  sampled_critiques: List[str],
                  sampled_scores: List[List[float]]) -> List[float]:
        num_responses = len(responses)
        num_samples = len(sampled_critiques)

        if num_samples == 0:
            print("Warning: No samples to aggregate. Returning zeros.")
            return [0.0] * num_responses
        if num_samples == 1:
            return sampled_scores[0] if sampled_scores and len(sampled_scores[0]) == num_responses else [0.0] * num_responses

        print(f"--- Aggregation (Strategy: {self.strategy}) ---")
        print(f"Aggregating {num_samples} samples for {num_responses} responses.")
        final_scores = [0.0] * num_responses
        current_strategy = self.strategy

        for attempt in range(2):
            if self.strategy == "simple_average":
                valid_samples_count = 0
                temp_final_scores = [0.0] * num_responses
                for i, scores_set in enumerate(sampled_scores):
                    if len(scores_set) == num_responses:
                        for j in range(num_responses): temp_final_scores[j] += scores_set[j]
                        valid_samples_count += 1
                    else: print(f"Warning: Skipping sample {i+1} in simple_average due to score count mismatch ({len(scores_set)} vs {num_responses})")
                if valid_samples_count > 0:
                    final_scores = [s / valid_samples_count for s in temp_final_scores]
                    break
                else:
                    print("Warning: No valid samples for simple_average. Returning zeros.")
                    final_scores = [0.0] * num_responses
                    break
            elif self.strategy == "meta_rm_guided_average":
                if not self.meta_rm_agent:
                    print("MetaRMAgent not initialized for 'meta_rm_guided_average' strategy. Falling back to simple_average.")
                    self.strategy = "simple_average"; continue
                weighted_scores_sum = [0.0] * num_responses
                total_meta_weight = 0.0
                valid_samples_for_meta_rm = 0
                for i in range(num_samples):
                    if len(sampled_scores[i]) != num_responses:
                        print(f"Warning: Skipping sample {i+1} in meta_rm_guided_average due to score count mismatch."); continue
                    current_principles_for_sample = sampled_principles[i] if i < len(sampled_principles) else sampled_principles[0]
                    meta_quality_score = self.meta_rm_agent.get_critique_quality_score(
                        query, responses, current_principles_for_sample, sampled_critiques[i], sampled_scores[i])
                    print(f"Sample {i+1} Meta-Quality: {meta_quality_score:.3f}, Scores: {sampled_scores[i]}")
                    for j in range(num_responses): weighted_scores_sum[j] += sampled_scores[i][j] * meta_quality_score
                    total_meta_weight += meta_quality_score
                    valid_samples_for_meta_rm +=1
                if total_meta_weight > 0:
                    final_scores = [ws / total_meta_weight for ws in weighted_scores_sum]; break
                elif valid_samples_for_meta_rm > 0 and total_meta_weight == 0:
                    print("Warning: Total meta weight is zero. Falling back to simple average over samples that had meta scores.")
                    self.strategy = "simple_average"; continue
                else:
                    print("Warning: Total meta weight is zero or no valid samples for meta_rm_guided_average. Falling back to simple average.")
                    self.strategy = "simple_average"; continue
            else:
                print(f"Unknown aggregation strategy: {self.strategy}. Defaulting to simple average.")
                self.strategy = "simple_average"; continue
        self.strategy = current_strategy
        print(f"Final Aggregated Scores: {[round(s, 2) for s in final_scores]}")
        return final_scores

class VerifierAgent(BaseAgent):
    """Verifies the MARM process for potential failures and adherence to principles."""
    def __init__(self, api_key: str, model_id: str = "gpt-4o-2024-05-13", temperature: float = 0.2): # Defaulting to an OpenAI model
        super().__init__("VerifierAgent", api_key, model_id, temperature)

    def verify_evaluation_process(self, query: str, responses: List[str],
                                  principles: str,
                                  critiques: List[str], # Note: This was `sampled_critiques` in the method signature
                                  aggregated_scores: List[float],
                                  full_trace_summary: str) -> str:
        prompt = f"""
        You are a meticulous Verification Agent. Review the entire multi-agent reward modeling process detailed below for potential issues and consistency.
        Focus on:
        1.  **Specification Adherence & Consistency (MAS Failure FM-1.1, FM-1.2):**
            * Are the generated `Principles` relevant and comprehensive for the `Query` and `Responses`?
            * Do the `Critiques` (see summary) consistently and logically apply these `Principles` to each response?
            * Do the `Aggregated Scores` appear to be a reasonable synthesis derived from the critiques and principles? (e.g., if critiques are harsh, scores should be low).
        2.  **Inter-Agent Misalignment & Reasoning Quality (MAS Failure FM-2.x, FM-3.x, JudgeLRM):**
            * If multiple critique samples were generated, were they wildly inconsistent in their assessment (indicating potential unreliability of critique agents)? (See full_trace_summary for clues).
            * Is there evidence of agents ignoring crucial information or context from the query or principles?
            * Is the reasoning within critiques sound and transparent? Do critiques justify their claims?
            * Does the aggregation process appear logical given the individual sample scores and any meta-assessment (if applicable)?

        Query:
        {query}

        Responses Provided (Summaries):
        {chr(10).join([f"- Response {i+1}: {r[:150].strip()}..." for i,r in enumerate(responses)])}

        Principles Used for Evaluation:
        {principles}

        Aggregated Scores: {aggregated_scores}

        Summary of Evaluation Process & Critiques (includes example critique content and scores):
        {full_trace_summary[:2000]}...

        Based on your review, provide a verification assessment.
        - If issues are found, identify specific concerns or potential failure modes (e.g., "Potential FM-2.6: Reasoning-action mismatch in critique for Response 1 as the critique was positive but score was very low.").
        - Conclude with 'Verification Passed.' or 'Verification Failed: [Briefly explain primary reason and identified concerns].'
        """
        verification_result = self.execute(prompt)
        return verification_result

class CoordinatorAgent:
    """
    Orchestrates the MARM workflow, managing interactions between specialized agents.
    """
    def __init__(self,
                 api_key: str, # This should be your OpenAI API key
                 num_critique_samples: int = 1,
                 aggregation_strategy: str = "meta_rm_guided_average",
                 model_ids: Optional[Dict[str, str]] = None,
                 use_dedicated_scoring_agent: bool = False):

        self.api_key = api_key
        self.use_dedicated_scoring_agent = use_dedicated_scoring_agent

        # Default to OpenAI models, but allow overrides
        default_models = {
            "principle": "gpt-4o-2024-05-13",  # OpenAI model
            "critique": "gpt-4o-2024-05-13",   # OpenAI model
            "scoring": "gpt-3.5-turbo",     # OpenAI model
            "meta_rm": "gpt-4o-2024-05-13",    # OpenAI model (or a cheaper one if suitable)
            "verifier": "gpt-4o-2024-05-13",  # OpenAI model
        }
        effective_model_ids = default_models.copy()
        if model_ids:
            effective_model_ids.update(model_ids)
            # Ensure all overridden models are also OpenAI compatible if only OpenAI key is used
            for role, model_id_val in effective_model_ids.items():
                if not model_id_val.startswith("gpt-"):
                    print(f"Warning: Model ID '{model_id_val}' for role '{role}' is overridden but may not be an OpenAI model. Ensure compatibility with the provided OpenAI API key.")


        self.principle_agent = PrincipleGeneratorAgent(api_key=self.api_key, model_id=effective_model_ids["principle"])
        self.critique_agents = [
            CritiqueAgent(api_key=self.api_key, model_id=effective_model_ids["critique"], temperature=round(random.uniform(0.6, 0.8),1))
            for _ in range(max(1, num_critique_samples))
        ]
        if self.use_dedicated_scoring_agent:
            self.scoring_agent = ScoringAgent(api_key=self.api_key, model_id=effective_model_ids["scoring"])
        else:
            self.scoring_agent = None

        self.aggregation_agent = AggregationAgent(api_key=self.api_key, strategy=aggregation_strategy, model_id_for_meta_rm=effective_model_ids["meta_rm"])
        self.verifier_agent = VerifierAgent(api_key=self.api_key, model_id=effective_model_ids["verifier"])
        self.num_critique_samples = max(1, num_critique_samples)

        print(f"Coordinator initialized with OpenAI API Key, {self.num_critique_samples} critique sample(s), '{aggregation_strategy}' strategy, Dedicated Scoring Agent: {self.use_dedicated_scoring_agent}.")
        print(f"Using models (ensure these are valid OpenAI model IDs if only OpenAI key is used): {effective_model_ids}")

    def get_multi_agent_reward(self, query: str, responses: List[str]) -> Dict[str, Any]:
        """Runs the full MARM pipeline."""
        print("\n=== MARM Pipeline Start ===")
        full_trace: Dict[str, Any] = {
            "query": query,
            "responses": responses,
            "source_file": "",           # 原始文件名
            "source_path": "",           # 完整的文件路径
            "generated_principles": "Not generated yet.",
            "sampled_critiques_with_scores": [],
            "aggregated_scores": [0.0] * len(responses),
            "verification_assessment": "Not performed yet.",
            "final_rewards_for_responses": [0.0] * len(responses)
        }

        print("\nStep 1: Generating Evaluation Principles...")
        try:
            principles = self.principle_agent.generate_principles(query, responses)
            full_trace["generated_principles"] = principles
            print(f"Principles Generated:\n{principles}")
        except Exception as e:
            print(f"Error in Principle Generation: {e}")
            full_trace["generated_principles"] = f"Error: {e}"; return full_trace

        print(f"\nStep 2: Generating {self.num_critique_samples} Critique Sample(s)...")
        sampled_principles_for_agg: List[str] = []
        sampled_critiques_for_agg: List[str] = []
        sampled_scores_for_agg: List[List[float]] = []

        for i in range(self.num_critique_samples):
            critique_agent = self.critique_agents[i]
            print(f"\n--- Critique Sample {i+1}/{self.num_critique_samples} (Agent: {critique_agent.model_id}, Temp: {critique_agent.temperature:.1f}) ---")
            try:
                critique_text, preliminary_scores = critique_agent.generate_critique_and_scores(query, responses, principles)
                if self.use_dedicated_scoring_agent and self.scoring_agent:
                    if not preliminary_scores or len(preliminary_scores) != len(responses) or any(s < 3 for s in preliminary_scores): # Heuristic to trigger refinement
                        print(f"Preliminary scores from CritiqueAgent for sample {i+1} ({preliminary_scores}) warrant refinement. Attempting with ScoringAgent.")
                        refined_scores = self.scoring_agent.get_scores_from_critique(query, responses, principles, critique_text)
                        if len(refined_scores) == len(responses):
                            print(f"ScoringAgent refined scores to: {refined_scores}"); preliminary_scores = refined_scores
                        else: print(f"ScoringAgent did not return valid scores ({refined_scores}). Sticking with CritiqueAgent scores.")
                
                full_trace["sampled_critiques_with_scores"].append({
                    "critique_agent_id": i, "critique_text": critique_text,
                    "preliminary_scores": preliminary_scores, "principles_used": principles})
                sampled_principles_for_agg.append(principles)
                sampled_critiques_for_agg.append(critique_text)
                sampled_scores_for_agg.append(preliminary_scores)
                print(f"Critique Sample {i+1} Text (first 300 chars):\n{critique_text[:300].strip()}...")
                print(f"Critique Sample {i+1} Preliminary Scores: {preliminary_scores}")
            except Exception as e:
                print(f"Error during Critique Sample {i+1} generation: {e}")
                full_trace["sampled_critiques_with_scores"].append({
                    "critique_agent_id": i, "error": str(e), "preliminary_scores": [0.0] * len(responses)})
                sampled_principles_for_agg.append(principles) # Still add principle if critique failed
                sampled_critiques_for_agg.append(f"Error: {e}")
                sampled_scores_for_agg.append([0.0] * len(responses)) # Add default scores for aggregation

        print("\nStep 3: Aggregating Scores...")
        if sampled_scores_for_agg:
            try:
                aggregated_scores = self.aggregation_agent.aggregate(
                    query, responses, sampled_principles_for_agg, sampled_critiques_for_agg, sampled_scores_for_agg)
                full_trace["aggregated_scores"] = [round(s, 2) for s in aggregated_scores]
                print(f"Final Aggregated Scores (raw): {aggregated_scores}")
            except Exception as e:
                print(f"Error during Aggregation: {e}")
                full_trace["aggregated_scores"] = [0.0] * len(responses); full_trace["aggregation_error"] = str(e)
        else:
            print("No valid scores to aggregate. Using default scores.")
            full_trace["aggregated_scores"] = [0.0] * len(responses)
        full_trace["final_rewards_for_responses"] = full_trace["aggregated_scores"]

        print("\nStep 4: Verifying Evaluation Process...")
        trace_summary_for_verifier = f"""
        Context: Evaluating {len(responses)} responses for the query: "{query[:100]}..."
        Principles: {full_trace['generated_principles'][:500]}...
        Number of Critique Samples: {len(full_trace['sampled_critiques_with_scores'])}
        Aggregation Strategy: {self.aggregation_agent.strategy}
        Dedicated Scoring Agent Used: {self.use_dedicated_scoring_agent}
        """
        if full_trace['sampled_critiques_with_scores']:
            valid_sample = next((s for s in full_trace['sampled_critiques_with_scores'] if 'error' not in s), None)
            if valid_sample:
                trace_summary_for_verifier += f"""
        Example Critique (Sample - Agent {valid_sample.get('critique_agent_id', 'N/A')}):
        Scores: {valid_sample.get('preliminary_scores', 'N/A')}
        Critique Text (start): {valid_sample.get('critique_text', 'N/A')[:500].strip()}..."""
            else: trace_summary_for_verifier += "\nNo valid critique samples were generated."
        if "meta_rm_guided_average" in self.aggregation_agent.strategy :
            trace_summary_for_verifier += "\nMeta-RM Guided Aggregation was the strategy."
        trace_summary_for_verifier += f"\nFinal Aggregated Scores: {full_trace['aggregated_scores']}"
        try:
            verification_assessment = self.verifier_agent.verify_evaluation_process(
                query, responses, full_trace["generated_principles"],
                sampled_critiques_for_agg, full_trace["aggregated_scores"], trace_summary_for_verifier)
            full_trace["verification_assessment"] = verification_assessment
            print(f"Verification Assessment:\n{verification_assessment}")
        except Exception as e:
            print(f"Error during Verification: {e}"); full_trace["verification_assessment"] = f"Error: {e}"
        print("\n=== MARM Pipeline End ===")
        return full_trace

# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing MARM System (Example 1 - OpenAI)...")
    # Set environment variables for OpenAI
    os.environ["OPENAI_API_KEY"] = "sk-sZxScCcTfpoDWTEBD6A4D8B00a544c2bB69bF14f8c358976"
    os.environ["OPENAI_BASE_URL"] = "https://api3.apifans.com/v1"
    
    from data_processor import read_ag2_data, save_marm_results
    
    # Define paths
    ag2_folder_path = "./AG2"
    results_output_path = "./results.json"
    
    # Use environment variable for API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Define which OpenAI models to use for each agent role
    openai_model_config = {
        "principle": "gpt-4o-2024-05-13",
        "critique": "gpt-4o-2024-05-13",
        "scoring": "gpt-3.5-turbo",
        "meta_rm": "gpt-4o-2024-05-13",
        "verifier": "gpt-4o-2024-05-13",
    }

    marm_system = CoordinatorAgent(
        api_key=OPENAI_API_KEY,
        num_critique_samples=1,
        aggregation_strategy="simple_average",
        model_ids=openai_model_config,
        use_dedicated_scoring_agent=False
    )

    # Read data from AG2 folder
    print("Reading data from AG2 folder...")
    data_list = read_ag2_data(ag2_folder_path)
    print(f"Found {len(data_list)} data entries")

    # Initialize wresults list and create initial results file
    results = []
    if not os.path.exists(results_output_path):
        save_marm_results(results, results_output_path)
    
    # Process each query-responses pair with progress bar
    with tqdm(total=len(data_list), desc="Processing samples", unit="sample") as pbar:
        for i, data in enumerate(data_list):
            pbar.set_description(f"Processing sample {i+1}/{len(data_list)}")
            
            results_trace = marm_system.get_multi_agent_reward(data['query'], data['responses'])
            results_trace['source_file'] = data.get('source_file', 'unknown')
            results_trace['source_path'] = os.path.abspath(os.path.join(ag2_folder_path, data.get('source_file', '')))
            results.append(results_trace)
            
            # Update results.json after each sample
            print(f"\nSaving results after sample {i+1}...")
            save_marm_results(results, results_output_path)
            print(f"Results saved to {results_output_path}")
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"Current file": data.get('source_file', 'unknown')})

    print("Done!")