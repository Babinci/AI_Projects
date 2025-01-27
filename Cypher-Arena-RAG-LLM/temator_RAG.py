import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import random
import os
import time
from tqdm import tqdm
from typing import Set, List


class FreestyleTopicGenerator:
    # def __init__(self, model_name="deepseek-r1:7b"):
    def __init__(self, model_name="mwiewior/bielik"):
        self.embeddings = HuggingFaceEmbeddings()
        self.llm = Ollama(
            model=model_name,
            temperature=0.9,  # Add creativity while maintaining coherence
        )
        # Load prompt template from file
        prompt_path = os.path.join("prompts", "topic_prompt.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            self.topic_prompt = PromptTemplate(
                input_variables=["context_batch", "similar_topics"], template=template
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt template file not found at {prompt_path}. Please create the file with your prompt template."
            )

    def load_data(self, articles_path, topics_path):
        """Load and prepare articles and topics data"""
        # Load articles with validation
        with open(articles_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        # Filter and process articles
        self.articles = [
            {"title": article.get("title", ""), "content": article.get("content", "")}
            for article in articles
            if isinstance(article, dict)
            and article.get("title")
            and article.get("content")
        ]

        print(f"Loaded {len(self.articles)} valid articles")

        # Load topics
        with open(topics_path, "r", encoding="utf-8") as f:
            self.topics = json.load(f)

        # Create embeddings for topics
        self.topics_vectorstore = FAISS.from_texts(self.topics, self.embeddings)

        # Create content chunks for articles (utilizing large context window)
        self.create_article_chunks()

    def create_article_chunks(self, chunk_size=50):
        """Create meaningful chunks of articles for context"""
        self.article_chunks = []
        current_chunk = []
        current_length = 0

        for article in self.articles:
            # Create concise summary of article
            article_summary = f"{article['title']}: {article['content'][:200]}..."

            if (
                current_length + len(article_summary) > 8000
            ):  # Conservative token estimate
                self.article_chunks.append(current_chunk)
                current_chunk = []
                current_length = 0

            current_chunk.append(article_summary)
            current_length += len(article_summary)

        if current_chunk:
            self.article_chunks.append(current_chunk)

        print(f"Created {len(self.article_chunks)} article chunks")

    def get_diverse_topics(self, k=30):
        """Get diverse existing topics for inspiration"""
        # Randomly sample from existing topics for diversity
        return random.sample(self.topics, min(k, len(self.topics)))

    def validate_topic(self, topic: str) -> bool:
        """Validate if topic meets our criteria"""
        # Basic validations using a validation map
        validation_rules = [
            # Rule function, error message (for debugging if needed)
            (lambda t: bool(t.strip()), "Empty topic"),
            (lambda t: 2 <= len(t.split()) <= 5, "Wrong word count"),
            (lambda t: not any(char.isdigit() for char in t), "Contains digits"),
            (lambda t: not t.isupper(), "All uppercase"),
            (lambda t: not any(char in t for char in '*:()[]'), "Contains special characters"),
            (
                lambda t: not any(
                    word.lower() in ['the', 'and', 'or', 'in', 'of', 'story', 'event', 'culture', 'vs'] 
                    for word in t.split()
                ),
                "Contains English words"
            ),
        ]
        
        return all(rule_func(topic.strip()) for rule_func, _ in validation_rules)

    def generate_topics(self, num_chunks=3, max_attempts=5):
        """Generate topics using multiple context chunks"""
        all_topics = []
        attempts = 0

        while len(all_topics) < 30 and attempts < max_attempts:
            # Select random chunks of articles
            selected_chunks = random.sample(
                self.article_chunks, min(num_chunks, len(self.article_chunks))
            )

            # Get diverse existing topics
            inspiration_topics = self.get_diverse_topics()

            for chunk in selected_chunks:
                # Combine articles in chunk
                context = "\n\n".join(chunk)

                # Create inspiration topics string
                similar_topics_str = "\n".join(
                    [f"- {topic}" for topic in inspiration_topics]
                )

                # Generate topics using this context
                chain = LLMChain(llm=self.llm, prompt=self.topic_prompt)
                result = chain.run(
                    {"context_batch": context, "similar_topics": similar_topics_str}
                )

                # Process results with careful dash removal
                new_topics = []
                for line in result.split("\n"):
                    line = line.strip()
                    if line.startswith("- "):  # Note the space after dash
                        new_topics.append(line[2:])  # Remove dash and space
                    elif line.startswith("-"):  # No space after dash
                        new_topics.append(line[1:])  # Remove just the dash

                # Extract actual topics and validate
                valid_topics = [
                    topic.strip()
                    for topic in new_topics
                    if self.validate_topic(topic.strip())
                ]

                all_topics.extend(valid_topics)

            attempts += 1

        # Remove duplicates while preserving order
        unique_topics = list(dict.fromkeys(all_topics))

        # Ensure we have exactly 30 topics
        if len(unique_topics) > 30:
            unique_topics = unique_topics[:30]

        return unique_topics


class TopicBatchGenerator:
    def __init__(self, base_generator: FreestyleTopicGenerator):
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


if __name__ == "__main__":
    # Initialize base generator
    base_generator = FreestyleTopicGenerator()

    # Load data
    print("Loading data...")
    base_generator.load_data(
        "/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/glamrap_articles_content_20250125_221305.json",
        "/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/temator_list.json",
    )

    # Initialize batch generator
    batch_generator = TopicBatchGenerator(base_generator)

    # Load existing topics for duplicate checking
    batch_generator.load_existing_topics(
        "/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/temator_list.json"
    )

    # Generate large dataset
    topics = batch_generator.generate_large_dataset(
        target_count=20000, batch_size=30, save_interval=30
    )
# if __name__ == "__main__":
#     import time

#     # Initialize generator
#     generator = FreestyleTopicGenerator()

#     # Load data
#     print("Loading data...")
#     generator.load_data(
#         '/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/glamrap_articles_content_20250125_221305.json',
#         '/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/temator_list.json'
#     )

#     # Generate topics
#     print("\nGenerating topics...")
#     start_time = time.time()
#     topics = generator.generate_topics(num_chunks=3)
#     end_time = time.time()

#     print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
#     print(f"Generated {len(topics)} unique topics")

#     # Save results
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     output_file = f'generated_topics_{timestamp}.json'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(topics, f, ensure_ascii=False, indent=2)

#     print(f"\nTopics saved to {output_file}")

#     # Print all topics
#     print("\nGenerated topics:")
#     for topic in topics:
#         print(f"- {topic}")
