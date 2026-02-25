"""Golden set management for RAG evaluation."""

import json
import jsonlines
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GoldenQAPair(BaseModel):
    """Schema for a golden QA pair."""

    id: str
    query: str
    ground_truth_answer: str
    ground_truth_contexts: List[str]
    metadata: Dict = Field(default_factory=dict)
    expected_sources: List[Dict] = Field(default_factory=list)


class GoldenSetManager:
    """Manage golden QA pairs for evaluation."""

    def __init__(self, data_dir: Path):
        """
        Initialize golden set manager.

        Args:
            data_dir: Directory for golden set storage
        """
        self.data_dir = data_dir
        self.golden_dir = data_dir / "golden_set"
        self.golden_dir.mkdir(parents=True, exist_ok=True)

    def load_golden_set(self, version: str = "current") -> List[GoldenQAPair]:
        """
        Load golden set from file.

        Args:
            version: Version identifier or 'current' for latest

        Returns:
            List of golden QA pairs
        """
        if version == "current":
            # Follow symlink or use latest version
            current_link = self.golden_dir / "current"
            if current_link.exists():
                golden_file = current_link / "golden_set.jsonl"
            else:
                # Find latest version
                versions = sorted([d for d in self.golden_dir.iterdir() if d.is_dir()])
                if not versions:
                    logger.warning("No golden set versions found")
                    return []
                golden_file = versions[-1] / "golden_set.jsonl"
        else:
            golden_file = self.golden_dir / version / "golden_set.jsonl"

        if not golden_file.exists():
            logger.warning(f"Golden set file not found: {golden_file}")
            return []

        qa_pairs = []
        with jsonlines.open(golden_file) as reader:
            for obj in reader:
                qa_pairs.append(GoldenQAPair(**obj))

        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {golden_file}")
        return qa_pairs

    def save_golden_set(
        self,
        qa_pairs: List[GoldenQAPair],
        version: str,
        changelog: Optional[str] = None
    ):
        """
        Save golden set to file.

        Args:
            qa_pairs: List of golden QA pairs
            version: Version identifier (e.g., 'v1.0')
            changelog: Optional changelog entry
        """
        version_dir = self.golden_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save QA pairs
        golden_file = version_dir / "golden_set.jsonl"
        with jsonlines.open(golden_file, mode='w') as writer:
            for qa in qa_pairs:
                writer.write(qa.model_dump())

        # Save metadata
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "total_pairs": len(qa_pairs),
            "query_type_distribution": self._compute_distribution(qa_pairs, "query_type"),
            "difficulty_distribution": self._compute_distribution(qa_pairs, "difficulty"),
        }
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save changelog
        if changelog:
            changelog_file = version_dir / "changelog.md"
            with open(changelog_file, 'w') as f:
                f.write(f"# Changelog - {version}\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
                f.write(changelog)

        logger.info(f"Saved {len(qa_pairs)} QA pairs to {golden_file}")

    def _compute_distribution(self, qa_pairs: List[GoldenQAPair], field: str) -> Dict:
        """Compute distribution of metadata field."""
        counts = {}
        for qa in qa_pairs:
            value = qa.metadata.get(field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    def validate_qa_pair(self, qa: GoldenQAPair) -> bool:
        """
        Validate a golden QA pair.

        Args:
            qa: QA pair to validate

        Returns:
            True if valid, False otherwise
        """
        checks = [
            (len(qa.query) > 10, "Query too short"),
            (len(qa.ground_truth_answer) > 20, "Answer too short"),
            (len(qa.ground_truth_contexts) > 0, "No ground truth contexts"),
            (qa.query.strip().endswith('?'), "Query should end with '?'"),
        ]

        for check, error_msg in checks:
            if not check:
                logger.warning(f"Validation failed for {qa.id}: {error_msg}")
                return False

        return True

    def add_qa_pair(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        metadata: Dict,
        sources: List[Dict],
        qa_id: Optional[str] = None
    ) -> GoldenQAPair:
        """
        Create and validate a new QA pair.

        Args:
            query: User query
            answer: Ground truth answer
            contexts: List of context IDs
            metadata: Metadata dictionary
            sources: Expected source documents
            qa_id: Optional QA pair ID

        Returns:
            Validated GoldenQAPair
        """
        if not qa_id:
            # Generate ID from query hash
            import hashlib
            qa_id = f"qa_{hashlib.md5(query.encode()).hexdigest()[:8]}"

        qa = GoldenQAPair(
            id=qa_id,
            query=query,
            ground_truth_answer=answer,
            ground_truth_contexts=contexts,
            metadata=metadata,
            expected_sources=sources
        )

        if not self.validate_qa_pair(qa):
            raise ValueError(f"QA pair validation failed: {qa_id}")

        return qa
