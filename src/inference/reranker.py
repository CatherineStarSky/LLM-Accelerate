#!/usr/bin/env python
"""LLM Re-ranking Inference Engine"""

import os
import time
import torch
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMReranker:
    """LLM-based re-ranking inference engine"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        use_simd: bool = False,
        batch_size: int = 1,
        simd_threads: int = 1,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize inference engine
        
        Args:
            model_name: Model name or path
            use_simd: Whether to enable SIMD optimization
            batch_size: Python-layer batch size (batch inference enabled when >=2)
            simd_threads: C++ SIMD operator thread count (only effective when use_simd=True)
            device: Device type
            verbose: Whether to print verbose information
        """
        self.model_name = model_name
        self.use_simd = use_simd
        self.batch_size = batch_size
        self.simd_threads = simd_threads
        self.device = device
        self.verbose = verbose
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        if self.verbose:
            print(f"[reranker] Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        # Note: If using SIMD optimization, recommend using float32 for best performance
        # Type conversion overhead for float16 models may offset SIMD benefits
        model_dtype = torch.float32 if self.use_simd else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
            device_map=None
        ).to(self.device)
        
        if self.verbose:
            print(f"[reranker] Model loaded on {self.device}")
        
        # Apply SIMD optimization
        if self.use_simd:
            if self.verbose:
                print("[reranker] Applying SIMD patches...")
            
            from .simd_patch import apply_simd_patches
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            apply_simd_patches(self.model, root_dir=root_dir, verbose=self.verbose)
            # Configure C++ SIMD thread count
            try:
                if self.simd_threads and self.simd_threads > 1:
                    torch.ops.simd_opt.set_threads(int(self.simd_threads))
                    if self.verbose:
                        print(f"[reranker] C++ SIMD threads set to {self.simd_threads}")
            except Exception as e:
                if self.verbose:
                    print(f"[reranker] Warning: set_threads failed: {e}")
        
        self.model.eval()
    
    def _create_prompt(self, context: str, candidate: str) -> str:
        """
        Create prompt
        
        Args:
            context: User history context
            candidate: Candidate description
        
        Returns:
            Complete prompt text
        """
        prompt = f"{context}\nCandidate: {candidate}\nIs this candidate relevant to the user? (yes/no)"
        return prompt
    
    def _score_single_candidate(self, prompt: str) -> float:
        """
        Score a single candidate
        
        Args:
            prompt: Complete prompt
        
        Returns:
            Relevance score (logits difference)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Get logits of the last token
            
            # Calculate yes/no logits difference as score
            yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
            
            score = (logits[yes_token_id] - logits[no_token_id]).item()
        
        return score

    def _score_batch(self, prompts: List[str]) -> List[float]:
        """
        Batch score multiple candidates (foundation for dynamic/static batch scheduling)
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_last = outputs.logits[:, -1, :]
            yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
            scores = (logits_last[:, yes_token_id] - logits_last[:, no_token_id]).tolist()
        return scores
    
    def rerank(
        self, 
        context: str, 
        candidates: List[Dict],
        return_time: bool = False
    ) -> Tuple[List[float], float]:
        """
        Re-rank candidates
        
        Args:
            context: User history context
            candidates: List of candidates, each containing 'text' field
            return_time: Whether to return inference time
        
        Returns:
            scores: Relevance scores for each candidate
            elapsed_time: Inference time (seconds)
        """
        start_time = time.time()
        
        if self.batch_size == 1:
            # Serial processing (no batching)
            scores = []
            for candidate in candidates:
                prompt = self._create_prompt(context, candidate['text'])
                score = self._score_single_candidate(prompt)
                scores.append(score)
        else:
            # Batch scheduler: use batch inference to improve throughput
            # Initial batch size specified by self.batch_size, can be adaptively adjusted
            batch_size = max(1, int(self.batch_size))
            scores = []
            prompts = [self._create_prompt(context, c['text']) for c in candidates]

            # Adaptive dynamic batch: adjust batch_size based on last batch elapsed time (cap at 32)
            adaptive = True
            last_elapsed = None
            i = 0
            while i < len(prompts):
                bs = batch_size
                batch = prompts[i:i+bs]
                t0 = time.time()
                batch_scores = self._score_batch(batch)
                t1 = time.time()
                scores.extend(batch_scores)
                last_elapsed = t1 - t0
                # Dynamic adjustment: simple heuristic
                if adaptive:
                    if last_elapsed > 1.0 and batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                    elif last_elapsed < 0.5:
                        batch_size = min(32, batch_size * 2)
                i += bs

        elapsed_time = time.time() - start_time
        
        if return_time:
            return scores, elapsed_time
        else:
            return scores
    
    def rerank_batch(
        self,
        samples: List[Dict],
        return_times: bool = False
    ) -> List[Tuple[List[float], float]]:
        """
        Batch process multiple samples
        
        Args:
            samples: List of samples, each containing 'context' and 'candidates'
            return_times: Whether to return inference time for each sample
        
        Returns:
            results: [(scores, time), ...] or [scores, ...]
        """
        results = []
        
        for sample in samples:
            scores, elapsed_time = self.rerank(
                sample['context'],
                sample['candidates'],
                return_time=True
            )
            
            if return_times:
                results.append((scores, elapsed_time))
            else:
                results.append(scores)
        
        return results


def main():
    """Test inference engine"""
    print("=== Testing LLM Reranker ===\n")
    
    # Create test samples
    context = "User liked movies: Toy Story, The Lion King, Aladdin"
    candidates = [
        {'text': 'Finding Nemo (Animation, Adventure)'},
        {'text': 'The Godfather (Crime, Drama)'},
        {'text': 'Shrek (Animation, Comedy)'},
    ]
    
    # Test without SIMD
    print("1. Testing without SIMD...")
    reranker_baseline = LLMReranker(use_simd=False, verbose=True)
    scores, time_baseline = reranker_baseline.rerank(context, candidates, return_time=True)
    print(f"Scores: {scores}")
    print(f"Time: {time_baseline:.3f}s\n")
    
    # Test with SIMD
    print("2. Testing with SIMD...")
    try:
        reranker_simd = LLMReranker(use_simd=True, verbose=True)
        scores, time_simd = reranker_simd.rerank(context, candidates, return_time=True)
        print(f"Scores: {scores}")
        print(f"Time: {time_simd:.3f}s")
        print(f"Speedup: {time_baseline/time_simd:.2f}x\n")
    except Exception as e:
        print(f"SIMD test failed: {e}\n")


if __name__ == "__main__":
    main()

