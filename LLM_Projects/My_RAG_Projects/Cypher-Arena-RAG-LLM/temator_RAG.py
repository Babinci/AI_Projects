import torch
from torch import amp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from temator_batch_generator import TopicBatchGenerator
import random
import json


class OptimizedFreestyleGenerator:
    def __init__(self, model_name="mwiewior/bielik"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="allegro/herbert-base-cased",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={
                "batch_size": 64,
                "normalize_embeddings": True,
            },
        )

        self.llm = Ollama(model=model_name, temperature=0.9)

        self.setup_prompts()
        self.articles_store_path = "vector_store/articles.faiss"
        self.topics_store_path = "vector_store/topics.faiss"

    def setup_prompts(self):
        """Load your original successful prompt"""
        print("Loading prompt templates...")
        with open("prompts/topic_prompt.txt", "r", encoding="utf-8") as f:
            template = f.read()
        self.topic_prompt = PromptTemplate(
            input_variables=["context_batch", "similar_topics"], template=template
        )

    def precompute_embeddings(self, articles_path: str, topics_path: str):
        """Precompute embeddings with optimized batch processing"""
        print(f"Starting embeddings precomputation...")
        print(f"Loading data from {articles_path} and {topics_path}")

        # Load data
        with open(articles_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        with open(topics_path, "r", encoding="utf-8") as f:
            topics = json.load(f)

        print(f"Loaded {len(articles)} articles and {len(topics)} topics")

        # Process articles in parallel
        print("Processing articles in parallel...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            processed_articles = list(executor.map(self._process_article, articles))
        print(f"Processed {len(processed_articles)} articles")

        # Create and save vector stores using GPU acceleration
        print("Creating vector stores...")
        with amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Articles store
            print("Creating articles vector store...")
            article_store = FAISS.from_texts(processed_articles, self.embeddings)
            article_store.save_local(self.articles_store_path)
            print(f"Articles vector store saved to {self.articles_store_path}")

            # Topics store
            print("Creating topics vector store...")
            topic_store = FAISS.from_texts(topics, self.embeddings)
            topic_store.save_local(self.topics_store_path)
            print(f"Topics vector store saved to {self.topics_store_path}")

        print("Embeddings precomputation completed")

    def _process_article(self, article: Dict) -> str:
        """Process single article"""
        if (
            isinstance(article, dict)
            and article.get("title")
            and article.get("content")
        ):
            return f"{article['title']}\n{article['content']}"
        return ""

    def load_vectors(self):
        """Load precomputed vector stores"""
        self.article_store = FAISS.load_local(
            self.articles_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.topic_store = FAISS.load_local(
            self.topics_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Topics vector store loaded")

    def generate_topics(self, num_chunks=3) -> List[str]:
        """Modified generation logic with more randomness and diversity"""

        # Get context using random sampling
        selected_chunks = []
        with amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Get larger pool of random articles
            all_docs = self.article_store.similarity_search("", k=100)
            # Sample num_chunks * 5 articles randomly
            selected_docs = random.sample(all_docs, k=num_chunks * 5)
            selected_chunks.extend([doc.page_content for doc in selected_docs])

        # Get more inspiration topics for diversity
        inspiration_topics = random.sample(
            [doc.page_content for doc in self.topic_store.similarity_search("", k=50)],
            k=15,  # Increased from 10 to 15
        )

        # Generate using your original method
        chain = LLMChain(llm=self.llm, prompt=self.topic_prompt)
        result = chain.run(
            {
                "context_batch": "\n\n".join(selected_chunks),
                "similar_topics": "\n".join(inspiration_topics),
            }
        )

        topics = []
        for line in result.split("\n"):
            topic = line.strip()
            if topic.startswith("- "):
                topic = topic[2:]
            if self.validate_topic(topic):
                topics.append(topic)
                if len(topics) >= 30:  # Stop once we have enough valid topics
                    break

        print(f"Generated {len(topics)} valid topics")
        return topics[:30]  # Ensure exactly 30 topics

    def validate_topic(self, topic: str) -> bool:
        """Your original validation logic"""
        if not topic:
            return False

        validation_rules = [
            (lambda t: bool(t.strip()), "Empty topic"),
            (lambda t: 2 <= len(t.split()) <= 5, "Wrong word count"),
            (lambda t: not any(char.isdigit() for char in t), "Contains digits"),
            (lambda t: not t.isupper(), "All uppercase"),
            (
                lambda t: not any(char in t for char in "*:()[]"),
                "Contains special characters",
            ),
            (lambda t: not t.startswith("<s>"), "Contains HTML tags"),
            (
                lambda t: not any(
                    word.lower()
                    in [
                        "the",
                        "and",
                        "or",
                        "in",
                        "of",
                        "story",
                        "event",
                        "culture",
                        "vs",
                    ]
                    for word in t.split()
                ),
                "Contains English words",
            ),
        ]

        return all(rule_func(topic.strip()) for rule_func, _ in validation_rules)


# Usage example
if __name__ == "__main__":
    print("Starting topic generation process...")
    generator = OptimizedFreestyleGenerator()

    # One-time preprocessing
    print("\nStarting preprocessing phase...")
    generator.precompute_embeddings(
        "temator_data/glamrap_articles_content_20250125_221305.json",
        "temator_data/temator_list.json",
    )

    # Load for generation
    print("\nLoading vector stores for generation...")
    generator.load_vectors()

    # Your existing batch generator can use this optimized generator
    print("\nInitializing batch generation...")
    batch_generator = TopicBatchGenerator(generator)
    batch_generator.load_existing_topics(
        "/home/wojtek/AI_Projects/Cypher-Arena-RAG-LLM/temator_data/temator_list.json"
    )
    print(f"Output will be saved to: {batch_generator.output_file}")

    print("\nStarting large dataset generation...")
    topics = batch_generator.generate_large_dataset(
        target_count=60000, batch_size=30, save_interval=30
    )
    print("Process completed successfully")
