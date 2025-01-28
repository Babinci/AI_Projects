import json
import os
import time
from tqdm import tqdm
from typing import Set, List

class TopicBatchGenerator:
    def __init__(self, base_generator):
        self.generator = base_generator
        self.existing_topics: Set[str] = set()
        self.generated_topics: Set[str] = set()
        self.output_file = "generated_topics/generated_topics.json"

    def load_existing_topics(self, filepath: str):
        """Load and store existing topics"""
        with open(filepath, "r", encoding="utf-8") as f:
            topics = json.load(f)
            self.existing_topics = {topic.lower() for topic in topics}
            print(f"Loaded {len(self.existing_topics)} existing topics")

    def load_progress(self) -> List[str]:
        """Load previously generated topics if they exist"""
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                previous_topics = json.load(f)
                # Add to tracking set
                for topic in previous_topics:
                    self.generated_topics.add(topic.lower())
                print(f"Loaded {len(previous_topics)} previously generated topics")
                return previous_topics
        except FileNotFoundError:
            print("No previous progress found, starting fresh")
            return []

    def is_unique_topic(self, topic: str) -> bool:
        """Check if topic is unique (case-insensitive)"""
        topic_lower = topic.lower()
        return (
            topic_lower not in self.existing_topics
            and topic_lower not in self.generated_topics
        )

    def _save_to_file(self, topics: List[str]):
        """Save/update topics to single JSON file"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)

    def generate_large_dataset(
        self, target_count: int, batch_size: int = 30, save_interval: int = 30
    ) -> List[str]:
        """Generate large dataset of unique topics with progress tracking and periodic saving"""
        # Load any existing progress
        result_topics = self.load_progress()
        start_time = time.time()

        # Adjust target based on already generated topics
        remaining_count = target_count - len(result_topics)
        if remaining_count <= 0:
            print(
                f"Already generated {len(result_topics)} topics, target was {target_count}"
            )
            return result_topics

        # Progress bar for remaining topics
        pbar = tqdm(total=remaining_count, desc="Generating topics")

        while len(result_topics) < target_count:
            # Generate a batch of topics
            new_topics = self.generator.generate_topics(num_chunks=3)

            # Filter unique topics
            unique_topics = [
                topic for topic in new_topics if self.is_unique_topic(topic)
            ]

            # Add unique topics to results and tracking set
            for topic in unique_topics:
                if len(result_topics) < target_count:
                    result_topics.append(topic)
                    self.generated_topics.add(topic.lower())
                    pbar.update(1)

                    # Save progress at intervals
                    if len(result_topics) % save_interval == 0:
                        self._save_to_file(result_topics)
                        print(
                            f"\nSaved {len(result_topics)} topics to {self.output_file}"
                        )
                else:
                    break

        pbar.close()

        # Save final results
        self._save_to_file(result_topics)

        end_time = time.time()
        print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
        print(f"Generated {len(result_topics)} unique topics")
        print(f"Results saved to {self.output_file}")

        return result_topics


# if __name__ == "__main__":
#     # Initialize base generator
#     base_generator = FreestyleTopicGenerator()

#     # Load data
#     print("Loading data...")
#     base_generator.load_data(
#         "/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/glamrap_articles_content_20250125_221305.json",
#         "/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/temator_list.json",
#     )

#     # Initialize batch generator
#     batch_generator = TopicBatchGenerator(base_generator)

#     # Load existing topics for duplicate checking
#     batch_generator.load_existing_topics(
#         "/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/temator_list.json"
#     )

#     # Generate large dataset
#     topics = batch_generator.generate_large_dataset(
#         target_count=80000, batch_size=30, save_interval=30
#     )

