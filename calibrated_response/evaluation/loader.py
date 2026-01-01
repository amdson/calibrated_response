"""Load evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from calibrated_response.models.question import MetaculusQuestion, Question


class DatasetLoader:
    """Load questions from various dataset formats."""
    
    def __init__(self, dataset_path: str | Path):
        """Initialize the loader.
        
        Args:
            dataset_path: Path to the dataset file (JSONL format expected)
        """
        self.dataset_path = Path(dataset_path)
    
    def load_all(self) -> list[MetaculusQuestion]:
        """Load all questions from the dataset.
        
        Returns:
            List of MetaculusQuestion objects
        """
        questions = []
        for q in self.iter_questions():
            questions.append(q)
        return questions
    
    def iter_questions(self) -> Iterator[MetaculusQuestion]:
        """Iterate over questions in the dataset.
        
        Yields:
            MetaculusQuestion objects
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    question = MetaculusQuestion.from_scraped_data(data)
                    yield question
                except Exception as e:
                    # Skip malformed entries
                    continue
    
    def load_sample(
        self,
        n: int = 10,
        resolved_only: bool = True,
        seed: Optional[int] = None,
    ) -> list[MetaculusQuestion]:
        """Load a random sample of questions.
        
        Args:
            n: Number of questions to sample
            resolved_only: Only include resolved questions
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled questions
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        all_questions = self.load_all()
        
        if resolved_only:
            all_questions = [q for q in all_questions if q.is_resolved()]
        
        if len(all_questions) <= n:
            return all_questions
        
        return random.sample(all_questions, n)
    
    def get_question_by_id(self, question_id: str | int) -> Optional[MetaculusQuestion]:
        """Get a specific question by ID.
        
        Args:
            question_id: Question ID to find
            
        Returns:
            Question if found, None otherwise
        """
        target_id = str(question_id)
        
        for q in self.iter_questions():
            if q.id == target_id or str(q.metaculus_id) == target_id:
                return q
        
        return None
    
    @staticmethod
    def from_scraped_file(path: str | Path) -> DatasetLoader:
        """Create a loader from the output of eval_scraper.py.
        
        Args:
            path: Path to questions.jsonl
            
        Returns:
            DatasetLoader instance
        """
        return DatasetLoader(path)
